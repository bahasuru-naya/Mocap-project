import sys
import queue
import multiprocessing as mp
from pathlib import Path
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.dashboard import Dashboard
from system.ble_manager import BLEThread
from system.cv_module import CVProcess
from system.sync_engine import StreamForwarder
from system.dataset_recorder import DatasetRecorder

def main():
    # Define the target devices (5 IMU sensors)
    # The firmware on each node MUST match one of these names.
    TARGET_DEVICES = ["ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3", "ESP32_IMU_4", "ESP32_IMU_5"]
    
    # Thread-safe queues for communication between modules
    ble_data_queue = queue.Queue(maxsize=5000)    # larger buffer for BLE burst handling
    ble_status_queue = queue.Queue()
    ble_cmd_queue = queue.Queue()
    
    # Cross-process queues (CV runs in its own process)
    cv_frame_queue = mp.Queue(maxsize=10)
    cv_data_queue = mp.Queue(maxsize=200)          # increased for smoother sync
    
    sync_queue = queue.Queue(maxsize=200)           # increased for dashboard smoothness

    # Initialize Components
    print("Initializing Mocap System...")
    recorder = DatasetRecorder(TARGET_DEVICES)

    # Initialize Threads
    ble_thread = BLEThread(TARGET_DEVICES, ble_data_queue, ble_status_queue, ble_cmd_queue)
    cv_process = CVProcess(cv_frame_queue, cv_data_queue)
    stream_forwarder = StreamForwarder(
        ble_data_queue, cv_data_queue, sync_queue, TARGET_DEVICES, recorder
    )

    # Start Threads
    print("Starting background threads...")
    ble_thread.start()
    cv_process.start()
    stream_forwarder.start()

    # Start UI
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Better look on Windows
    
    window = Dashboard(ble_cmd_queue, cv_frame_queue, sync_queue, ble_status_queue, recorder, TARGET_DEVICES)
    window.show()

    # Run Application Loop
    exit_code = app.exec_()

    # Clean shutdown
    print("Shutting down cleanly... Disconnecting BLE devices (this may take a few seconds)...")
    ble_thread.stop()
    cv_process.stop()
    
    # Crucial: Wait for the BLE thread to process the stop command and cleanly disconnect.
    # Without this, Windows BLE stack keeps the connections open, causing errors on next launch.
    ble_thread.join(timeout=5.0)
    cv_process.join(timeout=3.0)
    if cv_process.is_alive():
        cv_process.terminate()
    stream_forwarder.stop()
    stream_forwarder.join(timeout=2.0)
    recorder.stop_recording()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    mp.freeze_support()  # needed on Windows
    main()