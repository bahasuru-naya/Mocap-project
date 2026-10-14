"""
Convert the 3 trained ensemble members (Keras) to ONNX.

The notebook saves the ensemble as
    best_model_overall_member0.keras / _member1.keras / _member2.keras
This script converts every member to a matching .onnx file and checks that
each ONNX model gives the same output as its Keras original.

IMPORTANT: run it in an environment whose Keras version can load the .keras
files (the one you trained in, e.g. Kaggle), then copy the .onnx files to
prediction_system/model/.

Usage
-----
    cd prediction_system
    python src/convert_model_to_onnx.py
"""

import os
import sys
import glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_STEM = "best_model_overall"


# ---------------------------------------------------------------------- #
#  Custom layers used by the trained models. They must be registered with
#  the SAME names as in the notebook so load_model() can rebuild them.
#  At inference both are pass-through / constant, so ONNX export is clean.
# ---------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="mocap")
class SensorDropout(layers.Layer):
    def __init__(self, n_sensors, channels=6, rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_sensors, self.channels, self.rate = n_sensors, channels, rate

    def call(self, x, training=None):
        return x                                   # training-only layer -> identity

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


def convert_one(keras_path, onnx_path):
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
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(*spec_shape).astype("float32")
    y_keras = model(x, training=False).numpy()
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    diff = float(np.abs(y_keras - y_onnx).max())
    print(f"  saved {os.path.basename(onnx_path)} ({os.path.getsize(onnx_path)/1024:.1f} KB) | "
          f"max |keras - onnx| = {diff:.2e}")
    if diff > 1e-3:
        print("  [WARNING] difference is larger than expected - check the conversion!")
    return tuple(spec_shape)


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    members = sorted(glob.glob(os.path.join(MODEL_DIR, f"{MODEL_STEM}_member*.keras")))
    if not members:
        print(f"[ERROR] No '{MODEL_STEM}_member*.keras' files found in {MODEL_DIR}")
        sys.exit(1)

    print(f"Found {len(members)} ensemble member(s).")
    shapes = set()
    for path in members:
        print(f"\nConverting {os.path.basename(path)}")
        onnx_path = path.replace(".keras", ".onnx")
        shapes.add(convert_one(path, onnx_path))

    assert len(shapes) == 1, f"Members have different input shapes: {shapes}"
    print(f"\n[SUCCESS] {len(members)} ONNX model(s) written to {MODEL_DIR}  (input shape {shapes.pop()})")


if __name__ == "__main__":
    main()