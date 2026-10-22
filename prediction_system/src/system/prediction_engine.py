"""
PredictionEngine - real-time IMU -> 3-D pose inference thread.

Loads the self-contained ONNX model (best_model_overall_with_scalers.onnx).
The scalers are baked into the ONNX graph:
  raw IMU window -> [embedded scaler] -> [model ensemble] -> [embedded inverse scaler] -> pose

Runtime needs: onnxruntime, numpy, pandas (+ optional `onnx`, used only to read the
embedded scaler statistics for missing-sensor filling and out-of-range diagnostics).


"""

import os
import queue
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import onnxruntime as ort

try:                                   # optional: only used to read embedded scaler stats
    import onnx
    from onnx import numpy_helper
except ImportError:                    # pragma: no cover
    onnx = None
    numpy_helper = None


TARGET_DEVICES = [
    "ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3",
    "ESP32_IMU_4", "ESP32_IMU_5",
]

IMU_FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# 30 IMU input columns (5 nodes x 6 features) - MUST be the same order as training
IMU_COLS = [f"{dev}_{feat}" for dev in TARGET_DEVICES for feat in IMU_FEATURES]

# 51 pose output columns (17 keypoints x 3 axes)
POSE_COLS = [f"cv_{i}_{axis}" for i in range(17) for axis in ('x', 'y', 'z')]


# ---------------------------------------------------------------------- #
def fill_missing(X, fallback=None):
    """
    Fill NaN / inf gaps in a (window, features) array, column by column.

    - partial gaps: linear interpolation over the row index (leading/trailing gaps take the
      nearest valid value), same as the training pipeline's interpolate(method="linear")
    - column with no valid value at all: `fallback[j]` (the training mean of that column, so the
      standardised value is 0); if no fallback is available, 0.0.
    """
    X = np.array(X, dtype="float32", copy=True)
    X[~np.isfinite(X)] = np.nan
    nan_mask = np.isnan(X)
    if not nan_mask.any():
        return X

    idx = np.arange(X.shape[0])
    for j in np.where(nan_mask.any(axis=0))[0]:
        bad = nan_mask[:, j]
        if bad.all():
            X[:, j] = 0.0 if fallback is None else float(fallback[j])
        else:
            X[bad, j] = np.interp(idx[bad], idx[~bad], X[~bad, j])
    return X


# ====================================================================== #
class PosePredictor:
    """Loads the self-contained ONNX model with embedded scalers.

    Takes a raw IMU window (window_size, 30) and produces pose in original units (17, 3).
    """

    # initializer names created by fold_scalers_into_onnx() in the notebook
    _MEAN_NAME = "scaler_imu_mean"
    _SCALE_NAME = "scaler_imu_scale"

    def __init__(self, model_path, prefer_gpu=False, warn_z=8.0, **kwargs):
        # Support list or single path for backwards compatibility
        if isinstance(model_path, (list, tuple)):
            found = [p for p in model_path if "with_scalers" in os.path.basename(str(p))]
            model_path = found[0] if found else model_path[0]

        self.model_path = os.fspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if "with_scalers" not in os.path.basename(self.model_path):
            print("[PredictionEngine] WARNING: model file name has no 'with_scalers'. If this is the "
                  "plain best_model_overall.onnx it expects STANDARDISED input and returns STANDARDISED "
                  "output -> predictions will be wrong. Use best_model_overall_with_scalers.onnx.")

        providers = self._select_providers(prefer_gpu)
        self.session = self._create_session(self.model_path, providers)

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.window_size = int(shape[1]) if (len(shape) > 1 and isinstance(shape[1], int)) else 10
        self.feature_count = int(shape[2]) if (len(shape) > 2 and isinstance(shape[2], int)) else len(IMU_COLS)

        if self.feature_count != len(IMU_COLS):
            raise ValueError(
                f"Model expects {self.feature_count} features, pipeline provides {len(IMU_COLS)}"
            )

        model_output = self.session.get_outputs()[0]
        self.output_name = model_output.name

        # Embedded scaler statistics (used for missing-sensor fill + diagnostics)
        self.imu_mean, self.imu_scale = self._read_scaler_stats(self.model_path)
        self.warn_z = float(warn_z)
        self._last_warn = 0.0

        # Model warm-up
        dummy = np.zeros((1, self.window_size, self.feature_count), dtype="float32")
        self.session.run(None, {self.input_name: dummy})
        print(f"[PredictionEngine] Model warm-up complete. Window={self.window_size}, "
              f"Input='{self.input_name}', Output='{self.output_name}', "
              f"scaler stats={'yes' if self.imu_mean is not None else 'NO (install onnx)'}.")

    # ------------------------------------------------------------------ #
    @classmethod
    def _read_scaler_stats(cls, model_path):
        if onnx is None:
            print("[PredictionEngine] 'onnx' package not installed: missing sensors will be filled with 0 "
                  "(raw) instead of the training mean. `pip install onnx` for correct behaviour.")
            return None, None
        try:
            graph = onnx.load(model_path, load_external_data=False).graph
            stats = {t.name: numpy_helper.to_array(t) for t in graph.initializer
                     if t.name in (cls._MEAN_NAME, cls._SCALE_NAME)}
            mean = np.asarray(stats[cls._MEAN_NAME], dtype="float32").reshape(-1)
            scale = np.asarray(stats[cls._SCALE_NAME], dtype="float32").reshape(-1)
            if mean.size != len(IMU_COLS) or scale.size != len(IMU_COLS):
                raise ValueError(f"scaler size {mean.size}/{scale.size} != {len(IMU_COLS)}")
            scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
            return mean, scale
        except Exception as e:
            print(f"[PredictionEngine] Could not read embedded scaler stats ({e}); "
                  "missing sensors will be filled with 0 (raw).")
            return None, None

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
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers and providers[0] == 'DmlExecutionProvider':
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        print(f"[PredictionEngine] Loaded {os.path.basename(model_path)} "
              f"-> {session.get_providers()[0]}")
        return session

    # ------------------------------------------------------------------ #
    def _diagnose(self, X):
        """Throttled warning when the window is far outside the training distribution."""
        if self.imu_mean is None:
            return
        now = time.time()
        if now - self._last_warn < 2.0:
            return
        z = np.abs((X - self.imu_mean) / self.imu_scale)
        if z.max() > self.warn_z:
            col_max = z.max(axis=0)
            worst = np.argsort(col_max)[::-1][:3]
            desc = ", ".join(f"{IMU_COLS[j]} (|z|={col_max[j]:.1f})" for j in worst)
            print(f"[PredictionEngine] WARNING: input far outside training range -> {desc}. "
                  "Check units (g vs m/s^2, deg/s vs rad/s), axis order, device order, sensor offsets.")
            self._last_warn = now

    def predict_window(self, rows):
        """Predict one pose from an unscaled (window_size, 30) raw IMU window.

        Returns:
            np.ndarray: (17, 3) joint coordinates in original units.
        """
        X = np.asarray(rows, dtype="float32")
        expected_shape = (self.window_size, len(IMU_COLS))
        if X.shape != expected_shape:
            raise ValueError(f"Expected IMU window shape {expected_shape}, received {X.shape}")

        X = fill_missing(X, fallback=self.imu_mean)
        self._diagnose(X)
        pose = self.session.run(None, {self.input_name: np.ascontiguousarray(X[None], dtype="float32")})[0][0]
        return pose.reshape(17, 3)

    def predict_batch(self, windows):
        """Predict a batch of poses from unscaled raw IMU windows (N, window_size, 30)."""
        X = np.array(windows, dtype="float32", copy=True)
        if X.ndim != 3 or X.shape[1] != self.window_size or X.shape[2] != len(IMU_COLS):
            raise ValueError(
                f"Expected batch shape (N, {self.window_size}, {len(IMU_COLS)}), received {X.shape}"
            )
        for i in range(len(X)):
            X[i] = fill_missing(X[i], fallback=self.imu_mean)

        poses = self.session.run(None, {self.input_name: np.ascontiguousarray(X)})[0]
        return poses.reshape(-1, 17, 3)

    def predict_pose_from_imu(self, imu_window_df):
        """Predict one unscaled pose from a window of IMU rows in training column order."""
        missing_columns = [c for c in IMU_COLS if c not in imu_window_df.columns]
        if missing_columns:
            raise ValueError("IMU window is missing required columns: " + ", ".join(missing_columns))
        if len(imu_window_df) != self.window_size:
            raise ValueError(f"Expected {self.window_size} rows, got {len(imu_window_df)}")

        X_new = imu_window_df.loc[:, IMU_COLS].to_numpy(dtype="float32")
        pose = self.predict_window(X_new).reshape(-1)
        return pd.Series(pose, index=POSE_COLS)


# Backwards-compatibility alias
PoseEnsemble = PosePredictor


# ====================================================================== #
class PredictionEngine(threading.Thread):
    """
    Reads synchronised IMU frames from *sync_queue*, keeps a rolling window, and pushes
    predicted 3-D pose coordinates (17 x 3 array) into *pose_queue* for the dashboard.
    """

    def __init__(self, sync_queue, pose_queue, model_path=None,
                 imu_scaler_path=None, pose_scaler_path=None, prefer_gpu=False,
                 smoothing_alpha=0.0, **kwargs):
        super().__init__()
        self.sync_queue = sync_queue
        self.pose_queue = pose_queue
        self.running = True
        self.daemon = True
        self.prediction_enabled = False
        self._prediction_lock = threading.Lock()
        self._prediction_generation = 0

        # 0 = no smoothing; 0.3-0.6 = light EMA smoothing of the output pose
        self.smoothing_alpha = float(smoothing_alpha)
        self._smoothed = None

        if model_path is None:
            model_paths = kwargs.get("model_paths")
            if model_paths:
                model_path = model_paths
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(base_dir, "model", "best_model_overall_with_scalers.onnx")

        self.predictor = PosePredictor(model_path=model_path, prefer_gpu=prefer_gpu)
        self.window_size = self.predictor.window_size

        # Rolling window of IMU rows (float32 vectors of length 30, NaN = missing)
        self.window = deque(maxlen=self.window_size)

        # ----- FPS tracking ----- #
        self.inference_fps = 0
        self.inference_ms = 0.0
        self._ms_accum = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

        print(f"[PredictionEngine] Ready - model='{os.path.basename(self.predictor.model_path)}', "
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
                    if frames_consumed > 0 and len(self.window) == self.window_size
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
            self._smoothed = None
            self._prediction_generation += 1
            self.prediction_enabled = True

    def stop_prediction(self):
        with self._prediction_lock:
            self.prediction_enabled = False
            self._prediction_generation += 1
            self.window.clear()
            self._smoothed = None
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
        """SyncEngine frame -> float32 vector of the 30 IMU values (NaN where a sensor/value is missing)."""
        row = np.full(len(IMU_COLS), np.nan, dtype="float32")
        imu_all = frame.get('imu', {}) if isinstance(frame, dict) else {}
        for d, dev in enumerate(TARGET_DEVICES):
            imu = imu_all.get(dev)
            if imu is None:
                continue
            for f, feat in enumerate(IMU_FEATURES):
                try:
                    v = imu[feat]
                    if v is not None:
                        row[d * len(IMU_FEATURES) + f] = float(v)
                except (KeyError, TypeError, ValueError):
                    pass
        return row

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def _run_inference(self, generation, window_rows):
        try:
            t0 = time.perf_counter()
            # np.stack already creates a fresh array; avoid the redundant astype copy.
            raw_imu_window = np.stack(window_rows)

            # nothing usable at all in this window -> skip rather than predict garbage
            if not np.isfinite(raw_imu_window).any():
                return

            coords = self.predictor.predict_window(raw_imu_window).astype("float64")

            with self._prediction_lock:
                if not self.prediction_enabled or generation != self._prediction_generation:
                    return

                if self.smoothing_alpha > 0.0:
                    if self._smoothed is None:
                        self._smoothed = coords
                    else:
                        a = self.smoothing_alpha
                        self._smoothed = a * coords + (1.0 - a) * self._smoothed
                    coords = self._smoothed.copy()

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