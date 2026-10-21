"""
PredictionEngine — real-time IMU → 3-D pose inference thread.

Loads the self-contained ONNX model (best_model_overall_with_scalers.onnx).
The scalers are baked into the ONNX computational graph:
  raw IMU window -> [embedded scaler] -> [model ensemble] -> [embedded inverse scaler] -> pose

No scalers, no scikit-learn, no TensorFlow. Only `onnxruntime` and `numpy`:
  sess = ort.InferenceSession("best_model_overall_with_scalers.onnx")
  pose = sess.run(None, {"imu_raw": raw_imu_window.astype("float32")[None]})[0][0]
  joints = pose.reshape(17, 3)

The window length is read from the ONNX model itself (currently 10 frames).
"""

import os
import queue
import threading
import time
import numpy as np
import pandas as pd
from collections import deque
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


class PosePredictor:
    """Loads the self-contained ONNX model with embedded scalers.

    No external scalers, no scikit-learn, no TensorFlow needed at runtime.
    Takes raw IMU window (window_size, 30) and produces pose in original units (17, 3).
    """

    def __init__(self, model_path, prefer_gpu=False, **kwargs):
        # Support list or single path for backwards compatibility
        if isinstance(model_path, (list, tuple)):
            found = [p for p in model_path if "with_scalers" in os.path.basename(str(p))]
            model_path = found[0] if found else model_path[0]

        self.model_path = os.fspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        providers = self._select_providers(prefer_gpu)
        self.session = self._create_session(self.model_path, providers)

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.window_size = int(shape[1]) if (len(shape) > 1 and shape[1] is not None) else 10
        self.feature_count = int(shape[2]) if (len(shape) > 2 and shape[2] is not None) else len(IMU_COLS)

        if self.feature_count != len(IMU_COLS):
            raise ValueError(
                f"Model expects {self.feature_count} features, pipeline provides {len(IMU_COLS)}"
            )

        model_output = self.session.get_outputs()[0]
        self.output_name = model_output.name

        # Model warm-up
        dummy = np.zeros((1, self.window_size, self.feature_count), dtype="float32")
        self.session.run(None, {self.input_name: dummy})
        print(f"[PredictionEngine] Model warm-up complete. Window={self.window_size}, "
              f"Input='{self.input_name}', Output='{self.output_name}'.")

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
        """Predict one pose from an unscaled (window_size, 30) raw IMU window.

        Returns:
            np.ndarray: (17, 3) joint coordinates in original units.
        """
        X = np.asarray(rows, dtype="float32")
        expected_shape = (self.window_size, len(IMU_COLS))
        if X.shape != expected_shape:
            raise ValueError(f"Expected IMU window shape {expected_shape}, received {X.shape}")

        X = fill_missing(X.copy())
        # raw_imu_window: the last WINDOW_SIZE (10) raw IMU rows, shape (10, 30)
        # pose = sess.run(None, {"imu_raw": raw_imu_window.astype("float32")[None]})[0][0]
        # joints = pose.reshape(17, 3)
        pose = self.session.run(None, {self.input_name: X.astype("float32")[None]})[0][0]
        return pose.reshape(17, 3)

    def predict_batch(self, windows):
        """Predict a batch of poses from unscaled raw IMU windows.

        Args:
            windows: np.ndarray of shape (N, window_size, 30)
        Returns:
            np.ndarray: (N, 17, 3) joint coordinates in original units.
        """
        X = np.asarray(windows, dtype="float32")
        if X.ndim != 3 or X.shape[1] != self.window_size or X.shape[2] != len(IMU_COLS):
            raise ValueError(
                f"Expected batch shape (N, {self.window_size}, {len(IMU_COLS)}), received {X.shape}"
            )

        for i in range(len(X)):
            if np.isnan(X[i]).any():
                X[i] = fill_missing(X[i])

        poses = self.session.run(None, {self.input_name: X})[0]
        return poses.reshape(-1, 17, 3)

    def predict_pose_from_imu(self, imu_window_df):
        """Predict one unscaled pose from a window of IMU rows in training column order."""
        missing_columns = [column for column in IMU_COLS if column not in imu_window_df.columns]
        if missing_columns:
            raise ValueError(
                "IMU window is missing required columns: " + ", ".join(missing_columns)
            )
        if len(imu_window_df) != self.window_size:
            raise ValueError(
                f"Expected {self.window_size} rows, got {len(imu_window_df)}"
            )

        X_new = imu_window_df.loc[:, IMU_COLS].to_numpy(dtype="float32")
        pose = self.predict_window(X_new).reshape(-1)
        return pd.Series(pose, index=POSE_COLS)


# Backwards-compatibility alias
PoseEnsemble = PosePredictor


# ====================================================================== #
class PredictionEngine(threading.Thread):
    """
    Reads synchronised IMU frames from *sync_queue*, keeps a rolling
    window, and pushes predicted 3-D pose coordinates (17 × 3 array)
    into *pose_queue* for the dashboard to visualise.
    """

    def __init__(self, sync_queue, pose_queue, model_path=None,
                 imu_scaler_path=None, pose_scaler_path=None, prefer_gpu=False, **kwargs):
        super().__init__()
        self.sync_queue = sync_queue
        self.pose_queue = pose_queue
        self.running = True
        self.daemon = True
        self.prediction_enabled = False
        self._prediction_lock = threading.Lock()
        self._prediction_generation = 0

        if model_path is None:
            model_paths = kwargs.get("model_paths")
            if model_paths:
                model_path = model_paths
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(base_dir, "model", "best_model_overall_with_scalers.onnx")

        self.predictor = PosePredictor(
            model_path=model_path,
            prefer_gpu=prefer_gpu,
        )
        self.window_size = self.predictor.window_size

        # Rolling window of IMU row dicts
        self.window = deque(maxlen=self.window_size)

        # ----- FPS tracking ----- #
        self.inference_fps = 0
        self.inference_ms = 0.0          # mean time per prediction
        self._ms_accum = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

        print(f"[PredictionEngine] Ready — model='{os.path.basename(self.predictor.model_path)}', "
              f"window={self.window_size}, imu_features={len(IMU_COLS)}, "
              f"pose_outputs={len(POSE_COLS)}")

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            frames_consumed = 0

            with self._prediction_lock:
                generation = self._prediction_generation
                if self.prediction_enabled:
                    while True:
                        try:
                            frame = self.sync_queue.get_nowait()
                        except queue.Empty:
                            break
                        self.window.append(self._frame_to_row(frame))
                        frames_consumed += 1
                else:
                    self._drain_queue(self.sync_queue)
                    self.window.clear()

                inference_window = (
                    list(self.window)
                    if frames_consumed > 0
                    and len(self.window) == self.window_size
                    else None
                )

            if inference_window is not None:
                self._run_inference(generation, inference_window)

            time.sleep(0.005)

    @staticmethod
    def _drain_queue(target_queue):
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def start_prediction(self):
        with self._prediction_lock:
            self._drain_queue(self.sync_queue)
            self._drain_queue(self.pose_queue)
            self.window.clear()
            self._prediction_generation += 1
            self.prediction_enabled = True

    def stop_prediction(self):
        with self._prediction_lock:
            self.prediction_enabled = False
            self._prediction_generation += 1
            self.window.clear()
            self._drain_queue(self.sync_queue)
            self._drain_queue(self.pose_queue)
            self.inference_fps = 0
            self.inference_ms = 0.0
            self._frame_count = 0
            self._ms_accum = 0.0

    def is_prediction_enabled(self):
        with self._prediction_lock:
            return self.prediction_enabled

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
    def _run_inference(self, generation, window_rows):
        try:
            t0 = time.perf_counter()
            raw_imu_window = np.stack(window_rows)
            coords = self.predictor.predict_window(raw_imu_window)

            with self._prediction_lock:
                if (
                    not self.prediction_enabled
                    or generation != self._prediction_generation
                ):
                    return

                if self.pose_queue.full():
                    try:
                        self.pose_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.pose_queue.put({'coords': coords, 'timestamp': time.time()})

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
