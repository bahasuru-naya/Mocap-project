import sys
import queue
from PyQt5.QtWidgets import QApplication
from system.dashboard import Dashboard
from system.ble_manager import BLEThread
from system.cv_module import CVThread
from system.sync_engine import SyncEngine
from system.dataset_recorder import DatasetRecorder

def main():
    # Define the target devices (5 IMU sensors)
    # The firmware on each node MUST match one of these names.
    TARGET_DEVICES = ["ESP32_IMU_1", "ESP32_IMU_2", "ESP32_IMU_3", "ESP32_IMU_4", "ESP32_IMU_5"]
    
    # Thread-safe queues for communication between modules
    ble_data_queue = queue.Queue(maxsize=1000)
    ble_status_queue = queue.Queue()
    ble_cmd_queue = queue.Queue()
    
    cv_frame_queue = queue.Queue(maxsize=10)
    cv_data_queue = queue.Queue(maxsize=100)
    
    sync_queue = queue.Queue(maxsize=100)

    # Initialize Components
    print("Initializing Mocap System...")
    recorder = DatasetRecorder(TARGET_DEVICES)

    # Initialize Threads
    ble_thread = BLEThread(TARGET_DEVICES, ble_data_queue, ble_status_queue, ble_cmd_queue)
    cv_thread = CVThread(cv_frame_queue, cv_data_queue)
    sync_thread = SyncEngine(ble_data_queue, cv_data_queue, sync_queue, TARGET_DEVICES, recorder)

    # Start Threads
    print("Starting background threads...")
    ble_thread.start()
    cv_thread.start()
    sync_thread.start()

    # Start UI
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Better look on Windows
    
    window = Dashboard(ble_cmd_queue, cv_frame_queue, sync_queue, ble_status_queue, recorder, TARGET_DEVICES)
    window.show()

    # Run Application Loop
    exit_code = app.exec_()

    # Clean shutdown
    print("Shutting down cleanly...")
    ble_thread.stop()
    cv_thread.stop()
    sync_thread.stop()
    recorder.stop_recording()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
