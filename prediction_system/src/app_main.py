"""
AI MoCap — Realtime Motion Prediction System
=============================================

Entry point that wires together BLE sensor streaming, time-synchronisation,
ensemble pose inference (3 models averaged), and 3-D skeleton visualisation.

Usage
-----
    cd prediction_system
    python src/app_main.py
"""

import sys
import os
import queue

from PyQt5.QtWidgets import QApplication

from system.ble_manager import BLEThread
from system.sync_engine import SyncEngine
from system.prediction_engine import PredictionEngine
from system.dashboard import Dashboard


def main():
    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #
    TARGET_DEVICES = [
        "ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3",
        "ESP32_IMU_4", "ESP32_IMU_5",
    ]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The ensemble: 3 ONNX models (one per training seed), averaged at inference.
    # Inference device. Tiny models often run as fast (or faster) on CPU than on a GPU
    # because of host<->GPU copies. Flip this and compare the FPS / latency on the dashboard.
    PREFER_GPU = False

    N_MODELS = 3
    MODEL_PATHS = [
        os.path.join(BASE_DIR, "model", f"best_model_overall_member{i}.onnx")
        for i in range(N_MODELS)
    ]
    IMU_SCALER_PATH  = os.path.join(BASE_DIR, "scalers", "imu_scaler.joblib")
    POSE_SCALER_PATH = os.path.join(BASE_DIR, "scalers", "pose_scaler.joblib")

    missing = [p for p in MODEL_PATHS + [IMU_SCALER_PATH, POSE_SCALER_PATH] if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing file(s):")
        for p in missing:
            print("   ", p)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Thread-safe queues
    # ------------------------------------------------------------------ #
    ble_data_queue   = queue.Queue(maxsize=1000)  # BLE → SyncEngine
    ble_status_queue = queue.Queue()              # BLE → Dashboard (status)
    ble_cmd_queue    = queue.Queue()              # Dashboard → BLE (commands)
    sync_queue       = queue.Queue(maxsize=100)   # SyncEngine → PredictionEngine
    pose_queue       = queue.Queue(maxsize=10)    # PredictionEngine → Dashboard

    # ------------------------------------------------------------------ #
    #  Initialise components
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  AI MoCap — Realtime Motion Prediction System")
    print("=" * 60)

    print("\n[Main] Starting BLE thread...")
    ble_thread = BLEThread(
        TARGET_DEVICES, ble_data_queue, ble_status_queue, ble_cmd_queue
    )

    print("[Main] Starting sync engine...")
    sync_engine = SyncEngine(
        ble_data_queue, sync_queue, TARGET_DEVICES
    )

    print(f"[Main] Starting prediction engine (loading {N_MODELS} models)...")
    prediction_engine = PredictionEngine(
        sync_queue, pose_queue,
        MODEL_PATHS, IMU_SCALER_PATH, POSE_SCALER_PATH,
        prefer_gpu=PREFER_GPU,
    )

    # ------------------------------------------------------------------ #
    #  Start background threads
    # ------------------------------------------------------------------ #
    ble_thread.start()
    sync_engine.start()
    prediction_engine.start()

    # ------------------------------------------------------------------ #
    #  Launch Qt UI
    # ------------------------------------------------------------------ #
    print("\n[Main] Launching dashboard...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = Dashboard(
        ble_cmd_queue  = ble_cmd_queue,
        sync_queue     = sync_queue,
        pose_queue     = pose_queue,
        ble_status_queue = ble_status_queue,
        sync_engine    = sync_engine,
        prediction_engine = prediction_engine,
        target_devices = TARGET_DEVICES,
    )
    window.show()

    exit_code = app.exec_()

    # ------------------------------------------------------------------ #
    #  Clean shutdown
    # ------------------------------------------------------------------ #
    print("\n[Main] Shutting down... (disconnecting BLE devices)")
    ble_thread.stop()
    sync_engine.stop()
    prediction_engine.stop()

    ble_thread.join(timeout=5.0)
    sync_engine.join(timeout=2.0)
    prediction_engine.join(timeout=2.0)

    print("[Main] Shutdown complete.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()