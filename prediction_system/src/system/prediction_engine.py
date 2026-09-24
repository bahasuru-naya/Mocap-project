"""
PredictionEngine — real-time IMU → 3-D pose inference thread.

Uses ONNX Runtime with CUDA/GPU acceleration instead of TensorFlow,
avoiding DLL compatibility issues on Windows while providing fast inference.

Maintains a rolling window of synchronised IMU frames, applies time-aware
linear interpolation for missing values, and produces 17-joint 3-D pose
predictions.
"""

import threading
import time
import os
import numpy as np
import pandas as pd
from collections import deque
import joblib
import onnxruntime as ort


# ------------------------------------------------------------------ #
#  Constants — must match the training pipeline exactly
# ------------------------------------------------------------------ #
WINDOW_SIZE = 30       # Number of consecutive IMU frames per prediction

TARGET_DEVICES = [
    "ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3",
    "ESP32_IMU_4", "ESP32_IMU_5",
]

IMU_FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# 30 IMU input columns (5 nodes × 6 features)
IMU_COLS = [f"{dev}_{feat}" for dev in TARGET_DEVICES for feat in IMU_FEATURES]

# 51 pose output columns (17 keypoints × 3 axes)
POSE_COLS = [f"cv_{i}_{axis}" for i in range(17) for axis in ('x', 'y', 'z')]


# ====================================================================== #
class PredictionEngine(threading.Thread):
    """
    Reads synchronised IMU frames from *sync_queue*, keeps a rolling
    window, and pushes predicted 3-D pose coordinates (17 × 3 array)
    into *pose_queue* for the dashboard to visualise.
    """

    def __init__(self, sync_queue, pose_queue, model_path,
                 imu_scaler_path, pose_scaler_path):
        super().__init__()
        self.sync_queue = sync_queue
        self.pose_queue = pose_queue
        self.running = True
        self.daemon = True

        # Rolling window of IMU row dicts
        self.window = deque(maxlen=WINDOW_SIZE)

        # ----- Load scalers ----- #
        print(f"[PredictionEngine] Loading scalers...")
        self.imu_scaler = joblib.load(imu_scaler_path)
        self.pose_scaler = joblib.load(pose_scaler_path)

        # ----- Setup ONNX Runtime with GPU ----- #
        self.model_path = model_path
        self.session = self._create_ort_session(model_path)

        # Get input/output names from the ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ----- Warm-up inference (compiles execution graph) ----- #
        dummy = np.zeros((1, WINDOW_SIZE, len(IMU_COLS)), dtype="float32")
        self.session.run(None, {self.input_name: dummy})
        print("[PredictionEngine] Model warm-up complete.")

        # ----- FPS tracking ----- #
        self.inference_fps = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

        print(f"[PredictionEngine] Ready — window={WINDOW_SIZE}, "
              f"imu_features={len(IMU_COLS)}, pose_outputs={len(POSE_COLS)}")

    # ------------------------------------------------------------------ #
    #  ONNX Runtime session setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_ort_session(model_path):
        """
        Create an ONNX Runtime InferenceSession, preferring DirectML GPU
        (most reliable on Windows — no CUDA toolkit install needed),
        then CUDA, then falling back to CPU.
        """
        providers_available = ort.get_available_providers()
        print(f"[PredictionEngine] Available ONNX Runtime providers: {providers_available}")

        # Try GPU providers in order of preference
        if 'DmlExecutionProvider' in providers_available:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("[PredictionEngine] Using DirectML GPU for inference (GTX 1650).")
        elif 'CUDAExecutionProvider' in providers_available:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
                }),
                'CPUExecutionProvider',
            ]
            print("[PredictionEngine] Using CUDA GPU for inference.")
        else:
            providers = ['CPUExecutionProvider']
            print("[PredictionEngine] No GPU provider found — using CPU inference.")

        session = ort.InferenceSession(model_path, providers=providers)

        # Report which provider is actually being used
        active_providers = session.get_providers()
        print(f"[PredictionEngine] Active providers: {active_providers}")

        return session

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            frames_consumed = 0

            # Drain the sync queue into the rolling window
            while not self.sync_queue.empty():
                frame = self.sync_queue.get()
                row = self._frame_to_row(frame)
                self.window.append(row)
                frames_consumed += 1

            # Run inference once the window is full
            if frames_consumed > 0 and len(self.window) == WINDOW_SIZE:
                self._run_inference()

            # Yield CPU
            time.sleep(0.005)

    # ------------------------------------------------------------------ #
    #  Convert a synced frame dict into a flat row dict
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_to_row(frame):
        """
        Turn a SyncEngine frame into a flat dict with keys
        ``sys_time`` + each of the 30 ``IMU_COLS``.
        Missing sensors are stored as ``NaN``.
        """
        row = {'sys_time': frame['sys_time']}

        for dev in TARGET_DEVICES:
            imu = frame['imu'].get(dev)
            if imu is not None:
                for feat in IMU_FEATURES:
                    row[f"{dev}_{feat}"] = imu[feat]
            else:
                # Mark as NaN — will be interpolated before inference
                for feat in IMU_FEATURES:
                    row[f"{dev}_{feat}"] = float('nan')

        return row

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def _run_inference(self):
        try:
            # Build a DataFrame from the rolling window
            df = pd.DataFrame(list(self.window))

            # ----------------------------------------------------------
            # Missing values — interpolate only if any exist
            # (time-aware, since this is a time series)
            # ----------------------------------------------------------
            if df[IMU_COLS].isnull().sum().sum() > 0:
                df = df.set_index("sys_time")
                df[IMU_COLS] = df[IMU_COLS].interpolate(method="linear")
                df = df.reset_index()
                # Edge-fill any remaining NaN (first / last rows)
                df[IMU_COLS] = df[IMU_COLS].ffill().bfill()
                # Ultimate fallback — should never trigger in practice
                df[IMU_COLS] = df[IMU_COLS].fillna(0.0)

            # ---- Scale input ----------------------------------------- #
            X = df[IMU_COLS].values.astype("float32")            # (30, 30)
            X_scaled = self.imu_scaler.transform(X)
            X_batch = X_scaled[np.newaxis, ...].astype("float32")  # (1, 30, 30)

            # ---- GPU inference via ONNX Runtime ---------------------- #
            y_scaled = self.session.run(
                None, {self.input_name: X_batch}
            )[0]                                                  # (1, 51)

            # ---- Inverse-scale output -------------------------------- #
            y_pred = self.pose_scaler.inverse_transform(y_scaled)[0]  # (51,)

            # Reshape into 17 joints × 3 coordinates
            coords = y_pred.reshape(17, 3)

            # ---- Push to dashboard ----------------------------------- #
            if self.pose_queue.full():
                try:
                    self.pose_queue.get_nowait()
                except Exception:
                    pass
            self.pose_queue.put({
                'coords': coords,
                'timestamp': time.time(),
            })

            # ---- FPS tracking ---------------------------------------- #
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.inference_fps = self._frame_count
                self._frame_count = 0
                self._last_fps_time = now

        except Exception as e:
            print(f"[PredictionEngine] Inference error: {e}")

    # ------------------------------------------------------------------ #
    def stop(self):
        self.running = False
