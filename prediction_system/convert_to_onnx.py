"""
Convert the 3-member Keras ensemble + the two sklearn scalers into ONE self-contained ONNX file.

    raw IMU window (batch, 10, 30)
        -> (x - imu_mean) / imu_scale
        -> average of the 3 Keras members
        -> y * pose_scale + pose_mean
        -> pose in original units (batch, 51)

Input  (default folder ./data):
    ALL_best_model_overall_member0.keras
    ALL_best_model_overall_member1.keras
    ALL_best_model_overall_member2.keras
    ALL_imu_scaler.joblib
    ALL_pose_scaler.joblib
Output (default folder ./model):
    best_model_overall_with_scalers.onnx      (input "imu_raw", output "pose_original_units")

Usage:  python convert_to_onnx.py [--data-dir data] [--model-dir model] [--prefix ALL_]
"""
import argparse
import copy
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Force CPU *before* TensorFlow is imported -> the traced graph never contains cuDNN-only ops
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from onnx import helper, numpy_helper
from tensorflow import keras
from tensorflow.keras import layers

try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable

OUTPUT_NAME = "best_model_overall_with_scalers.onnx"
RNN_NAMES = ("GRU", "LSTM")
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# Custom layer used while training ("mocap>SensorDropout"). It is training-only, so at
# inference it is the identity. It is re-declared here only so Keras can deserialise the files.
# --------------------------------------------------------------------------------------
@register_keras_serializable(package="mocap", name="SensorDropout")
class SensorDropout(layers.Layer):
    def __init__(self, n_sensors=5, channels=6, rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_sensors = n_sensors
        self.channels = channels
        self.rate = rate

    def call(self, x, training=None):
        return x                                   # inference: identity

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(n_sensors=self.n_sensors, channels=self.channels, rate=self.rate)
        return cfg


CUSTOM_OBJECTS = {"SensorDropout": SensorDropout, "mocap>SensorDropout": SensorDropout}


# --------------------------------------------------------------------------------------
# Locate input files
# --------------------------------------------------------------------------------------
def find_inputs(data_dir, prefix):
    members = glob.glob(os.path.join(data_dir, f"{prefix}best_model_overall_member*.keras"))
    members.sort(key=lambda p: int(re.search(r"member(\d+)", os.path.basename(p)).group(1)))
    if not members:
        raise FileNotFoundError(f"No '{prefix}best_model_overall_member*.keras' files in {data_dir}")

    def one(pattern):
        hits = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not hits:
            raise FileNotFoundError(f"'{pattern}' not found in {data_dir}")
        return hits[0]

    return members, one(f"{prefix}imu_scaler.joblib"), one(f"{prefix}pose_scaler.joblib")


# --------------------------------------------------------------------------------------
# Merge ensemble into ONE Keras graph (same logic as notebook section 19.2)
# --------------------------------------------------------------------------------------
def _walk(layer):
    yield layer
    for sub in getattr(layer, "layers", []):
        yield from _walk(sub)


def _no_cudnn_clone(m):
    def fn(layer):
        cfg = layer.get_config()
        if type(layer).__name__ in RNN_NAMES:
            cfg["use_cudnn"] = False
        return layer.__class__.from_config(cfg)
    c = keras.models.clone_model(m, clone_function=fn)
    c.set_weights(m.get_weights())
    return c


def ensemble_to_single_keras(members, window, n_feat, rebuild=False):
    inp = keras.Input(shape=(window, n_feat), name="imu_window")
    outs = []
    for i, m in enumerate(members):
        if rebuild:
            m = _no_cudnn_clone(m)
        sub = keras.Model(m.inputs, m.outputs[0], name=f"member{i}")   # unique names
        outs.append(sub(inp, training=False))
    out = outs[0] if len(outs) == 1 else layers.Average(name="ensemble_mean")(outs)
    return keras.Model(inp, out, name="ensemble_single_graph")


def disable_cudnn(model):
    n = 0
    for lyr in _walk(model):
        if type(lyr).__name__ in RNN_NAMES:
            lyr.use_cudnn = False
            n += 1
    print(f"  disabled cuDNN on {n} recurrent layer(s)")


def make_serve(model, spec):
    @tf.function(input_signature=list(spec))
    def _serve(x):
        return model(x, training=False)
    return _serve


def cudnn_ops_left(serve_fn):
    with tf.device("/CPU:0"):
        gd = serve_fn.get_concrete_function().graph.as_graph_def()
    ops = {n.op for n in gd.node}
    for f in gd.library.function:
        ops |= {n.op for n in f.node_def}
    return sorted(o for o in ops if "Cudnn" in o)


# --------------------------------------------------------------------------------------
# Keras -> ONNX (standardised in / standardised out), newest opset that converts AND validates
# --------------------------------------------------------------------------------------
def convert_to_onnx(single_model, serve, spec, out_path, tmp_dir):
    def validate():
        onnx.checker.check_model(onnx.load(out_path))

    def try_from_function(opset):
        with tf.device("/CPU:0"):
            tf2onnx.convert.from_function(serve, input_signature=spec, opset=opset, output_path=out_path)
        validate()

    def try_saved_model_cli(opset):
        sm_dir = os.path.join(tmp_dir, "saved_model")
        shutil.rmtree(sm_dir, ignore_errors=True)
        single_model.export(sm_dir)
        subprocess.check_call([sys.executable, "-m", "tf2onnx.convert", "--saved-model", sm_dir,
                               "--output", out_path, "--opset", str(opset)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        validate()

    last_err = None
    for strategy in (try_from_function, try_saved_model_cli):
        for opset in (17, 15, 13):
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
                strategy(opset)
                print(f"  converted with {strategy.__name__}, opset {opset}")
                return
            except Exception as e:                      # try next opset / strategy
                last_err = e
    raise RuntimeError(f"ONNX conversion failed with every strategy. Last error: {last_err!r}")


# --------------------------------------------------------------------------------------
# Fold the scalers into the ONNX graph (same logic as notebook section 19.4a)
# --------------------------------------------------------------------------------------
def fold_scalers_into_onnx(src_path, dst_path, imu_scaler, pose_scaler,
                           new_input="imu_raw", new_output="pose_original_units"):
    m = onnx.load(src_path)
    g = m.graph

    init_names = {t.name for t in g.initializer}
    real_inputs = [i for i in g.input if i.name not in init_names]
    assert len(real_inputs) == 1 and len(g.output) == 1, "expected exactly one input and one output"
    old_in_vi, old_out_vi = real_inputs[0], g.output[0]
    old_in, old_out = old_in_vi.name, old_out_vi.name
    scaled_in, scaled_out = old_in + "__scaled", old_out + "__scaled"

    for n in g.node:
        for k, name in enumerate(n.input):
            if name == old_in:
                n.input[k] = scaled_in
        for k, name in enumerate(n.output):
            if name == old_out:
                n.output[k] = scaled_out

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    g.initializer.extend([
        numpy_helper.from_array(f32(imu_scaler.mean_),   "scaler_imu_mean"),
        numpy_helper.from_array(f32(imu_scaler.scale_),  "scaler_imu_scale"),
        numpy_helper.from_array(f32(pose_scaler.scale_), "scaler_pose_scale"),
        numpy_helper.from_array(f32(pose_scaler.mean_),  "scaler_pose_mean"),
    ])

    pre = [helper.make_node("Sub", [new_input, "scaler_imu_mean"], [new_input + "__centered"], name="imu_center"),
           helper.make_node("Div", [new_input + "__centered", "scaler_imu_scale"], [scaled_in], name="imu_scale")]
    post = [helper.make_node("Mul", [scaled_out, "scaler_pose_scale"], [new_output + "__mul"], name="pose_unscale"),
            helper.make_node("Add", [new_output + "__mul", "scaler_pose_mean"], [new_output], name="pose_uncenter")]

    nodes = pre + list(g.node) + post
    g.ClearField("node")
    g.node.extend(nodes)

    new_in_vi, new_out_vi = copy.deepcopy(old_in_vi), copy.deepcopy(old_out_vi)
    new_in_vi.name, new_out_vi.name = new_input, new_output
    g.input.remove(old_in_vi); g.input.insert(0, new_in_vi)
    g.output.remove(old_out_vi); g.output.append(new_out_vi)

    for vi in [vi for vi in g.value_info if vi.name in (old_in, old_out)]:
        g.value_info.remove(vi)

    onnx.checker.check_model(m)
    onnx.save(m, dst_path)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "model"))
    ap.add_argument("--prefix", default="ALL_", help="file-name prefix of the input files (default: ALL_)")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    final_path = os.path.join(args.model_dir, OUTPUT_NAME)

    print("[1/5] Locating input files")
    member_paths, imu_path, pose_path = find_inputs(args.data_dir, args.prefix)
    for p in member_paths + [imu_path, pose_path]:
        print("  ", p)

    print("[2/5] Loading Keras members and scalers")
    members = [keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS, compile=False) for p in member_paths]
    imu_scaler, pose_scaler = joblib.load(imu_path), joblib.load(pose_path)
    window, n_feat = members[0].input_shape[1:3]
    n_out = members[0].output_shape[-1]
    assert imu_scaler.mean_.shape[0] == n_feat, f"imu scaler has {imu_scaler.mean_.shape[0]} features, model expects {n_feat}"
    assert pose_scaler.mean_.shape[0] == n_out, f"pose scaler has {pose_scaler.mean_.shape[0]} outputs, model has {n_out}"
    print(f"   {len(members)} members | window={window} | features={n_feat} | outputs={n_out}")

    print("[3/5] Merging the ensemble into one graph")
    spec = (tf.TensorSpec((None, window, n_feat), tf.float32, name="imu_window"),)
    single = ensemble_to_single_keras(members, window, n_feat)
    disable_cudnn(single)
    serve = make_serve(single, spec)
    left = cudnn_ops_left(serve)
    if left:
        print("   cuDNN ops found, rebuilding members without cuDNN ...")
        single = ensemble_to_single_keras(members, window, n_feat, rebuild=True)
        disable_cudnn(single)
        serve = make_serve(single, spec)
        left = cudnn_ops_left(serve)
    if left:
        raise RuntimeError(f"cuDNN ops still present in the graph: {left}")

    rng = np.random.default_rng(0)
    raw_check = (rng.standard_normal((256, window, n_feat)) * imu_scaler.scale_ + imu_scaler.mean_).astype("float32")
    scaled_check = ((raw_check - imu_scaler.mean_) / imu_scaler.scale_).astype("float32")
    ens_scaled = np.mean([m.predict(scaled_check, verbose=0) for m in members], axis=0)
    diff = np.abs(single.predict(scaled_check, verbose=0) - ens_scaled).max()
    print(f"   merged graph vs member average: max |diff| = {diff:.2e}")
    assert diff < 1e-4, "merged graph does not match the ensemble average"

    print("[4/5] Converting to ONNX and folding the scalers in")
    with tempfile.TemporaryDirectory() as tmp:
        scaled_onnx = os.path.join(tmp, "best_model_overall.onnx")
        convert_to_onnx(single, serve, spec, scaled_onnx, tmp)
        fold_scalers_into_onnx(scaled_onnx, final_path, imu_scaler, pose_scaler)

    print("[5/5] Verifying the saved file")
    proto = onnx.load(final_path)
    onnx.checker.check_model(proto)
    sess = ort.InferenceSession(final_path, providers=["CPUExecutionProvider"])
    onnx_pose = sess.run(None, {sess.get_inputs()[0].name: raw_check})[0]
    ref_pose = pose_scaler.inverse_transform(ens_scaled)
    d = np.abs(onnx_pose - ref_pose).max()
    print(f"   ONNX (raw in) vs Keras ensemble + sklearn scalers: max |diff| = {d:.2e}")
    print("   input :", [(i.name, [x.dim_value or x.dim_param for x in i.type.tensor_type.shape.dim]) for i in proto.graph.input])
    print("   output:", [(o.name, [x.dim_value or x.dim_param for x in o.type.tensor_type.shape.dim]) for o in proto.graph.output])
    if d > 1e-3:
        raise RuntimeError("Parity check FAILED - inspect the conversion")
    print(f"\nDONE -> {final_path}  ({os.path.getsize(final_path) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
