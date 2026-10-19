"""
Convert the trained ensemble to ONNX: the 3 Keras members AND the 2 scalers.

Inputs (in prediction_system/):
    model/best_model_overall_member0.keras .. member2.keras
    scalers/imu_scaler.joblib
    scalers/pose_scaler.joblib

Outputs (written next to the inputs):
    model/best_model_overall_member{i}.onnx
    scalers/imu_scaler.onnx          y = (x - mean) / scale
    scalers/pose_scaler.onnx         y = x * scale + mean   (inverse transform)

The prediction system loads only the .onnx files, so it needs neither
TensorFlow, scikit-learn nor joblib at runtime. The .keras and .joblib files
are kept as the conversion source.

Every output is checked against its original (Keras model / sklearn scaler),
and the full scaler -> model -> inverse-scaler chain is checked end to end.

IMPORTANT: model conversion must run in an environment whose Keras version can
load the .keras files (the one you trained in). Scaler conversion needs only
numpy, onnx, onnxruntime, scikit-learn and joblib.

Usage
-----
    cd prediction_system
    python src/convert_model_to_onnx.py                  # models + scalers
    python src/convert_model_to_onnx.py --scalers-only   # no TensorFlow needed
    python src/convert_model_to_onnx.py --models-only
"""

import argparse
import glob
import os
import sys

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_STEM = "best_model_overall"
SCALER_NAMES = ("imu_scaler", "pose_scaler")
ONNX_OPSET_SCALER = 13
ONNX_IR_VERSION = 8          # readable by onnxruntime 1.14 and newer
SCALER_TOLERANCE = 1e-4
MODEL_TOLERANCE = 1e-3


# ====================================================================== #
#  Scalers: sklearn StandardScaler -> ONNX
# ====================================================================== #
def _scaler_params(scaler):
    """Return (mean, scale) as float32 vectors, honouring with_mean / with_std."""
    n_features = int(scaler.n_features_in_)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    mean = np.zeros(n_features) if mean is None else np.asarray(mean)
    scale = np.ones(n_features) if scale is None else np.asarray(scale)
    return mean.astype("float32"), scale.astype("float32")


def _build_affine_model(graph_name, ops, n_features):
    """Build an ONNX graph that applies [(op_type, constant_vector), ...] in order."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["N", n_features]
    )
    graph_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["N", n_features]
    )

    initializers, nodes, previous = [], [], "input"
    for index, (op_type, constant) in enumerate(ops):
        constant_name = f"const_{index}"
        output_name = "output" if index == len(ops) - 1 else f"tmp_{index}"
        initializers.append(numpy_helper.from_array(constant, constant_name))
        nodes.append(helper.make_node(op_type, [previous, constant_name], [output_name]))
        previous = output_name

    graph = helper.make_graph(
        nodes, graph_name, [graph_input], [graph_output], initializer=initializers
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET_SCALER)]
    )
    model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model)
    return model


def convert_scaler(joblib_path, onnx_path):
    """Convert one StandardScaler. imu_scaler = transform, pose_scaler = inverse."""
    import joblib
    import onnx
    import onnxruntime as ort

    scaler = joblib.load(joblib_path)
    mean, scale = _scaler_params(scaler)
    n_features = mean.shape[0]
    is_pose = os.path.basename(joblib_path).startswith("pose")

    if is_pose:                                   # inverse_transform: x * scale + mean
        ops = [("Mul", scale), ("Add", mean)]
        reference = scaler.inverse_transform
        name = "pose_scaler_inverse"
    else:                                         # transform: (x - mean) / scale
        ops = [("Sub", mean), ("Div", scale)]
        reference = scaler.transform
        name = "imu_scaler_transform"

    model = _build_affine_model(name, ops, n_features)
    onnx.save(model, onnx_path)

    # ---- verify against sklearn ---- #
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (mean + rng.standard_normal((256, n_features)) * scale).astype("float32")
    expected = reference(x.astype("float64"))
    actual = session.run(None, {"input": x})[0]
    diff = float(np.abs(expected - actual).max())
    print(f"  saved {os.path.basename(onnx_path)} ({os.path.getsize(onnx_path)} B, "
          f"{n_features} features) | max |sklearn - onnx| = {diff:.2e}")
    if diff > SCALER_TOLERANCE:
        print("  [WARNING] difference is larger than expected - check the conversion!")
    return n_features


def convert_scalers(scaler_dir):
    missing = [
        n for n in SCALER_NAMES
        if not os.path.isfile(os.path.join(scaler_dir, f"{n}.joblib"))
    ]
    if missing:
        print(f"[ERROR] Missing scaler file(s) in {scaler_dir}: "
              + ", ".join(f"{n}.joblib" for n in missing))
        sys.exit(1)

    for scaler_name in SCALER_NAMES:
        print(f"\nConverting {scaler_name}.joblib")
        convert_scaler(
            os.path.join(scaler_dir, f"{scaler_name}.joblib"),
            os.path.join(scaler_dir, f"{scaler_name}.onnx"),
        )


# ====================================================================== #
#  Models: Keras -> ONNX
# ====================================================================== #
def _register_custom_layers(tf, keras, layers):
    """
    Custom layers used by the trained models. They must be registered with the
    SAME names as in the notebook so load_model() can rebuild them. At inference
    both are pass-through / constant, so ONNX export is clean.
    """
    @keras.utils.register_keras_serializable(package="mocap")
    class SensorDropout(layers.Layer):
        def __init__(self, n_sensors, channels=6, rate=0.05, **kwargs):
            super().__init__(**kwargs)
            self.n_sensors, self.channels, self.rate = n_sensors, channels, rate

        def call(self, x, training=None):
            return x                               # training-only layer -> identity

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            cfg = super().get_config()
            cfg.update(n_sensors=self.n_sensors, channels=self.channels, rate=self.rate)
            return cfg

    @keras.utils.register_keras_serializable(package="mocap")
    class PositionalEncoding(layers.Layer):
        def __init__(self, seq_len, d_model, **kwargs):
            super().__init__(**kwargs)
            self.seq_len, self.d_model = seq_len, d_model
            pos = np.arange(seq_len)[:, None]
            i = np.arange(d_model)[None, :]
            angle_rads = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            self.pos_encoding = tf.constant(angle_rads[None, ...], dtype=tf.float32)

        def call(self, x):
            return x + tf.cast(self.pos_encoding, x.dtype)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            cfg = super().get_config()
            cfg.update(seq_len=self.seq_len, d_model=self.d_model)
            return cfg


def convert_one_model(tf, keras, keras_path, onnx_path):
    import tf2onnx
    import onnxruntime as ort

    model = keras.models.load_model(keras_path, compile=False)
    in_shape = model.input_shape                       # (None, WINDOW, 30)
    spec_shape = [1 if d is None else d for d in in_shape]
    print(f"  input {in_shape}  output {model.output_shape}")

    spec = [tf.TensorSpec(spec_shape, tf.float32, name="input")]
    try:
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=onnx_path)
    except Exception as e:                             # Keras 3 fallback
        print(f"  from_keras failed ({type(e).__name__}); retrying via tf.function")
        fn = tf.function(lambda x: model(x, training=False), input_signature=spec)
        tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=onnx_path)

    # ---- verify: ONNX output must match Keras output ---- #
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(*spec_shape).astype("float32")
    y_keras = model(x, training=False).numpy()
    y_onnx = session.run(None, {session.get_inputs()[0].name: x})[0]
    diff = float(np.abs(y_keras - y_onnx).max())
    print(f"  saved {os.path.basename(onnx_path)} ({os.path.getsize(onnx_path)/1024:.1f} KB) | "
          f"max |keras - onnx| = {diff:.2e}")
    if diff > MODEL_TOLERANCE:
        print("  [WARNING] difference is larger than expected - check the conversion!")
    return tuple(spec_shape)


def convert_models(model_dir):
    members = sorted(glob.glob(os.path.join(model_dir, f"{MODEL_STEM}_member*.keras")))
    if not members:
        print(f"[ERROR] No '{MODEL_STEM}_member*.keras' files found in {model_dir}")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _register_custom_layers(tf, keras, layers)

    print(f"Found {len(members)} ensemble member(s).")
    shapes = set()
    for path in members:
        print(f"\nConverting {os.path.basename(path)}")
        onnx_path = path[:-len(".keras")] + ".onnx"
        shapes.add(convert_one_model(tf, keras, path, onnx_path))

    assert len(shapes) == 1, f"Members have different input shapes: {shapes}"
    print(f"\n{len(members)} ONNX model(s) written to {model_dir}  (input shape {shapes.pop()})")


# ====================================================================== #
#  End-to-end check of the ONNX-only pipeline
# ====================================================================== #
def verify_pipeline(model_dir, scaler_dir):
    """ONNX scaler -> ONNX models -> ONNX inverse scaler  vs  the joblib scalers."""
    import joblib
    import onnxruntime as ort

    model_paths = sorted(glob.glob(os.path.join(model_dir, f"{MODEL_STEM}_member*.onnx")))
    if not model_paths:
        return
    cpu = ["CPUExecutionProvider"]
    imu_ref = joblib.load(os.path.join(scaler_dir, "imu_scaler.joblib"))
    pose_ref = joblib.load(os.path.join(scaler_dir, "pose_scaler.joblib"))
    imu_onnx = ort.InferenceSession(os.path.join(scaler_dir, "imu_scaler.onnx"), providers=cpu)
    pose_onnx = ort.InferenceSession(os.path.join(scaler_dir, "pose_scaler.onnx"), providers=cpu)
    sessions = [ort.InferenceSession(p, providers=cpu) for p in model_paths]

    window = sessions[0].get_inputs()[0].shape[1]
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        raw = (imu_ref.mean_ + rng.standard_normal((window, imu_ref.n_features_in_))
               * imu_ref.scale_).astype("float32")
        scaled_ref = imu_ref.transform(raw)[None].astype("float32")
        scaled_onnx = imu_onnx.run(None, {"input": raw})[0][None]
        out_ref = np.mean([s.run(None, {s.get_inputs()[0].name: scaled_ref})[0]
                           for s in sessions], axis=0)
        out_onnx = np.mean([s.run(None, {s.get_inputs()[0].name: scaled_onnx})[0]
                            for s in sessions], axis=0)
        pose_a = pose_ref.inverse_transform(out_ref)
        pose_b = pose_onnx.run(None, {"input": out_onnx.astype("float32")})[0]
        worst = max(worst, float(np.abs(pose_a - pose_b).max()))
    print(f"\nEnd-to-end check (joblib scalers vs ONNX scalers, same models): "
          f"max |pose difference| = {worst:.2e}")
    if worst > 1e-4:
        print("[WARNING] end-to-end difference is larger than expected!")


def main():
    parser = argparse.ArgumentParser(description="Convert models and scalers to ONNX.")
    parser.add_argument("--models-only", action="store_true", help="skip the scalers")
    parser.add_argument("--scalers-only", action="store_true",
                        help="skip the models (TensorFlow not required)")
    args = parser.parse_args()
    if args.models_only and args.scalers_only:
        parser.error("Choose at most one of --models-only / --scalers-only.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    scaler_dir = os.path.join(base_dir, "scalers")

    if not args.scalers_only:
        convert_models(model_dir)
    if not args.models_only:
        convert_scalers(scaler_dir)
        verify_pipeline(model_dir, scaler_dir)

    print("\n[SUCCESS] Conversion finished.")


if __name__ == "__main__":
    main()
