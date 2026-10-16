"""
PredictionEngine — real-time IMU → 3-D pose inference thread.

Loads an ENSEMBLE of ONNX models (the 3 training seeds), runs each on the same
input window and averages their outputs, exactly like EnsembleModel.predict()
in the training notebook.

The window length is read from the ONNX model itself, so it always matches the
training pipeline (currently 10 frames).
"""

import os
import queue
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


class PoseEnsemble:
    """Loads the trained ensemble and applies its shared IMU-to-pose pipeline."""

    def __init__(self, model_paths, imu_scaler_path, pose_scaler_path, prefer_gpu=True):
        print("[PredictionEngine] Loading scalers...")
        self.imu_scaler = joblib.load(imu_scaler_path)
        self.pose_scaler = joblib.load(pose_scaler_path)
        if self.imu_scaler.n_features_in_ != len(IMU_COLS):
            raise ValueError(
                f"IMU scaler expects {self.imu_scaler.n_features_in_} features, "
                f"pipeline provides {len(IMU_COLS)}"
            )
        if self.pose_scaler.n_features_in_ != len(POSE_COLS):
            raise ValueError(
                f"Pose scaler expects {self.pose_scaler.n_features_in_} features, "
                f"pipeline provides {len(POSE_COLS)}"
            )

        providers = self._select_providers(prefer_gpu)
        self.sessions = [self._create_session(path, providers) for path in model_paths]
        self.input_names = [session.get_inputs()[0].name for session in self.sessions]

        input_shapes = {tuple(session.get_inputs()[0].shape[1:]) for session in self.sessions}
        if len(input_shapes) != 1:
            raise ValueError(f"Ensemble members have different input shapes: {input_shapes}")
        self.window_size, feature_count = input_shapes.pop()
        if feature_count != len(IMU_COLS):
            raise ValueError(
                f"Model expects {feature_count} features, pipeline provides {len(IMU_COLS)}"
            )

        output_shapes = {tuple(session.get_outputs()[0].shape[1:]) for session in self.sessions}
        if len(output_shapes) != 1:
            raise ValueError(f"Ensemble members have different output shapes: {output_shapes}")
        output_shape = output_shapes.pop()
        if int(np.prod(output_shape)) != len(POSE_COLS):
            raise ValueError(
                f"Model outputs {output_shape}, pipeline expects {len(POSE_COLS)} pose values"
            )

        dummy = np.zeros((1, self.window_size, len(IMU_COLS)), dtype="float32")
        for session, input_name in zip(self.sessions, self.input_names):
            session.run(None, {input_name: dummy})
        print("[PredictionEngine] Model warm-up complete.")

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
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        print(f"[PredictionEngine] Loaded {os.path.basename(model_path)} "
              f"-> {session.get_providers()[0]}")
        return session

    def predict_window(self, rows):
        """Predict one pose from an unscaled (window_size, 30) IMU window."""
        X = np.asarray(rows, dtype="float32")
        expected_shape = (self.window_size, len(IMU_COLS))
        if X.shape != expected_shape:
            raise ValueError(f"Expected IMU window shape {expected_shape}, received {X.shape}")

        X = fill_missing(X.copy())
        X_batch = self.imu_scaler.transform(X)[np.newaxis, ...].astype("float32")
        outputs = [
            session.run(None, {input_name: X_batch})[0]
            for session, input_name in zip(self.sessions, self.input_names)
        ]
        y_scaled = np.mean(outputs, axis=0)
        y_pred = self.pose_scaler.inverse_transform(y_scaled)[0]
        return y_pred.reshape(17, 3)


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
        self.predictor = PoseEnsemble(
            self.model_paths,
            imu_scaler_path,
            pose_scaler_path,
            prefer_gpu=prefer_gpu,
        )
        self.window_size = self.predictor.window_size

        # Rolling window of IMU row dicts
        self.window = deque(maxlen=self.window_size)

        # ----- FPS tracking ----- #
        self.inference_fps = 0
        self.inference_ms = 0.0          # mean time per prediction (all ensemble members)
        self._ms_accum = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

        print(f"[PredictionEngine] Ready — ensemble={len(self.predictor.sessions)} models, "
              f"window={self.window_size}, imu_features={len(IMU_COLS)}, "
              f"pose_outputs={len(POSE_COLS)}")

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
            coords = self.predictor.predict_window(np.stack(self.window))

            # ---- Push to dashboard ----------------------------------- #
            if self.pose_queue.full():
                try:
                    self.pose_queue.get_nowait()
                except queue.Empty:
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