"""
Convert the trained Keras model to ONNX format for use with ONNX Runtime.

This avoids TensorFlow DLL compatibility issues on Windows and provides
faster, lighter inference via onnxruntime-gpu (DirectML / CUDA).

Usage
-----
    cd prediction_system
    python src/convert_model_to_onnx.py
"""

import os
import sys
import numpy as np

# Suppress TF warnings during conversion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KERAS_PATH = os.path.join(BASE_DIR, "model", "best_model_overall.keras")
    ONNX_PATH  = os.path.join(BASE_DIR, "model", "best_model_overall.onnx")

    if not os.path.exists(KERAS_PATH):
        print(f"[ERROR] Keras model not found: {KERAS_PATH}")
        sys.exit(1)

    print(f"Loading Keras model: {KERAS_PATH}")

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(KERAS_PATH)

    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Convert to ONNX using tf2onnx
    import tf2onnx

    # Build a concrete input spec from the model
    input_shape = model.input_shape  # e.g. (None, 30, 30)
    # Replace None batch dim with 1 for the spec
    spec_shape = [1 if d is None else d for d in input_shape]

    input_spec = [tf.TensorSpec(spec_shape, tf.float32, name="input")]

    print(f"\nConverting to ONNX (opset 17)...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=17,
        output_path=ONNX_PATH,
    )

    print(f"\n[SUCCESS] ONNX model saved: {ONNX_PATH}")
    print(f"  File size: {os.path.getsize(ONNX_PATH) / 1024:.1f} KB")

    # Verify the ONNX model runs
    print("\nVerifying ONNX model with dummy input...")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros(spec_shape, dtype=np.float32)
    result = sess.run(None, {input_name: dummy})
    print(f"  Output shape: {result[0].shape}")
    print(f"  Verification PASSED.")


if __name__ == "__main__":
    main()
