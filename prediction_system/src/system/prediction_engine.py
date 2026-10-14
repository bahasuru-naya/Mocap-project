"""
PredictionEngine — real-time IMU → 3-D pose inference thread.

Loads an ENSEMBLE of ONNX models (the 3 training seeds), runs each on the same
input window and averages their outputs, exactly like EnsembleModel.predict()
in the training notebook.

The window length is read from the ONNX model itself, so it always matches the
training pipeline (currently 10 frames).
"""

import os
import threading
import time
import numpy as np
from collections import deque
import joblib
import onnxruntime as ort


TARGET_DEVICES = [
    "ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3",
    "ESP32_IMU_4", "ESP32_IMU_5",
]

IMU_FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# 30 IMU input columns (5 nodes × 6 features) — same order as training
IMU_COLS = [f"{dev}_{feat}" for dev in TARGET_DEVICES for feat in IMU_FEATURES]

# 51 pose output columns (17 keypoints × 3 axes)
POSE_COLS = [f"cv_{i}_{axis}" for i in range(17) for axis in ('x', 'y', 'z')]


def fill_missing(X):
    """
    Fill NaN gaps in a (window, features) array, column by column.
    Linear interpolation over the row index (same as the training pipeline's
    interpolate(method="linear")); leading/trailing gaps take the nearest valid
    value; a column with no valid value at all becomes 0.
    """
    nan_mask = np.isnan(X)
    if not nan_mask.any():
        return X
    idx = np.arange(X.shape[0])
    for j in np.where(nan_mask.any(axis=0))[0]:
        bad = nan_mask[:, j]
        if bad.all():
            X[:, j] = 0.0
        else:
            X[bad, j] = np.interp(idx[bad], idx[~bad], X[~bad, j])
    return X


# ====================================================================== #
class PredictionEngine(threading.Thread):
    """
    Reads synchronised IMU frames from *sync_queue*, keeps a rolling
    window, and pushes predicted 3-D pose coordinates (17 × 3 array)
    into *pose_queue* for the dashboard to visualise.
    """

    def __init__(self, sync_queue, pose_queue, model_paths,
                 imu_scaler_path, pose_scaler_path, prefer_gpu=True):
        super().__init__()
        self.sync_queue = sync_queue
        self.pose_queue = pose_queue
        self.running = True
        self.daemon = True

        if isinstance(model_paths, (str, bytes)):      # single path still works
            model_paths = [model_paths]
        self.model_paths = list(model_paths)

        # ----- Load scalers ----- #
        print("[PredictionEngine] Loading scalers...")
        self.imu_scaler = joblib.load(imu_scaler_path)
        self.pose_scaler = joblib.load(pose_scaler_path)

        # ----- Load every ensemble member ----- #
        providers = self._select_providers(prefer_gpu)
        self.sessions = [self._create_session(p, providers) for p in self.model_paths]
        self.input_names = [s.get_inputs()[0].name for s in self.sessions]

        # Window length comes from the model, not from a hard-coded constant
        shapes = {tuple(s.get_inputs()[0].shape[1:]) for s in self.sessions}
        if len(shapes) != 1:
            raise ValueError(f"Ensemble members have different input shapes: {shapes}")
        self.window_size, n_feat = shapes.pop()
        if n_feat != len(IMU_COLS):
            raise ValueError(f"Model expects {n_feat} features, pipeline provides {len(IMU_COLS)}")

        # Rolling window of IMU row dicts
        self.window = deque(maxlen=self.window_size)

        # ----- Warm-up inference ----- #
        dummy = np.zeros((1, self.window_size, len(IMU_COLS)), dtype="float32")
        for sess, name in zip(self.sessions, self.input_names):
            sess.run(None, {name: dummy})
        print("[PredictionEngine] Model warm-up complete.")

        # ----- FPS tracking ----- #
        self.inference_fps = 0
        self.inference_ms = 0.0          # mean time per prediction (all ensemble members)
        self._ms_accum = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

        print(f"[PredictionEngine] Ready — ensemble={len(self.sessions)} models, "
              f"window={self.window_size}, imu_features={len(IMU_COLS)}, "
              f"pose_outputs={len(POSE_COLS)}")

    # ------------------------------------------------------------------ #
    #  ONNX Runtime setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_providers(prefer_gpu):
        available = ort.get_available_providers()
        print(f"[PredictionEngine] Available ONNX Runtime providers: {available}")

        if prefer_gpu and 'DmlExecutionProvider' in available:
            print("[PredictionEngine] Using DirectML GPU for inference.")
            return ['DmlExecutionProvider', 'CPUExecutionProvider']
        if prefer_gpu and 'CUDAExecutionProvider' in available:
            print("[PredictionEngine] Using CUDA GPU for inference.")
            return [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
                }),
                'CPUExecutionProvider',
            ]
        print("[PredictionEngine] Using CPU inference.")
        return ['CPUExecutionProvider']

    @staticmethod
    def _create_session(model_path, providers):
        opts = ort.SessionOptions()
        if providers and providers[0] == 'DmlExecutionProvider':
            # Recommended by ONNX Runtime for the DirectML provider
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        print(f"[PredictionEngine] Loaded {os.path.basename(model_path)} "
              f"-> {session.get_providers()[0]}")
        return session

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            frames_consumed = 0

            while not self.sync_queue.empty():
                frame = self.sync_queue.get()
                self.window.append(self._frame_to_row(frame))
                frames_consumed += 1

            if frames_consumed > 0 and len(self.window) == self.window_size:
                self._run_inference()

            time.sleep(0.005)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_to_row(frame):
        """SyncEngine frame -> float32 vector of the 30 IMU values (NaN where a sensor is missing)."""
        row = np.full(len(IMU_COLS), np.nan, dtype="float32")
        for d, dev in enumerate(TARGET_DEVICES):
            imu = frame['imu'].get(dev)
            if imu is not None:
                for f, feat in enumerate(IMU_FEATURES):
                    row[d * len(IMU_FEATURES) + f] = imu[feat]
        return row

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def _run_inference(self):
        try:
            t0 = time.perf_counter()

            X = np.stack(self.window)                                  # (W, 30)
            X = fill_missing(X)                                        # interpolate dropped packets
            X_batch = self.imu_scaler.transform(X)[np.newaxis, ...].astype("float32")

            # ---- Run every ensemble member and average ---------------- #
            outputs = [
                sess.run(None, {name: X_batch})[0]                     # (1, 51)
                for sess, name in zip(self.sessions, self.input_names)
            ]
            y_scaled = np.mean(outputs, axis=0)

            # ---- Inverse-scale output -------------------------------- #
            y_pred = self.pose_scaler.inverse_transform(y_scaled)[0]   # (51,)
            coords = y_pred.reshape(17, 3)

            # ---- Push to dashboard ----------------------------------- #
            if self.pose_queue.full():
                try:
                    self.pose_queue.get_nowait()
                except Exception:
                    pass
            self.pose_queue.put({'coords': coords, 'timestamp': time.time()})

            # ---- FPS / latency tracking ------------------------------ #
            self._frame_count += 1
            self._ms_accum += (time.perf_counter() - t0) * 1000.0
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.inference_fps = self._frame_count
                self.inference_ms = self._ms_accum / max(self._frame_count, 1)
                self._frame_count = 0
                self._ms_accum = 0.0
                self._last_fps_time = now

        except Exception as e:
            print(f"[PredictionEngine] Inference error: {e}")

    # ------------------------------------------------------------------ #
    def stop(self):
        self.running = False