# AI MoCap Data Collection System

A complete data collection system for synchronized AI Motion Capture, utilizing 5 BLE-enabled ESP32 + BMI160 nodes and a MediaPipe-based Computer Vision module.

## Setup Instructions

### Hardware Setup (Sensor Nodes)
1. You will need 5 ESP32 microcontrollers and 5 BMI160 sensors.
2. The sensors should be wired via I2C to the ESP32 (default pins used in `IMU_sensor_node_bluetooth.ino`).
3. Open `IMU_sensor_node_bluetooth.ino` in the Arduino IDE.
4. **CRITICAL:** For each node, change the `#define NODE_ID` line to give it a unique ID before uploading. You MUST use the names expected by the Python script:
   - `ESP32_IMU_1`
   - `ESP32_IMU_2`
   - `ESP32_IMU_3`
   - `ESP32_IMU_4`
   - `ESP32_IMU_5`
5. Upload the code to all 5 ESP32 modules.

### PC Software Setup
1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Connect a webcam to your PC.
4. Turn on your ESP32 modules.

### Running the System
Run the main application:
```bash
python src/app_main.py
```

### Dashboard Usage
- **Calibration:** Before starting a recording, place all IMUs flat and stationary, and click "Calibrate All Sensors". Wait for the status indicator on each node to turn Blue ("CALIBRATED"). This will cancel out gyroscope drift and zero out the accelerometer offset.
- **Recording:** Enter a session name and click "Start Recording".
- **Synchronization Monitor:** Ensure the "Sync Rate" is stable (usually between 20-30 FPS).
- **Data Output:** Recorded datasets are saved in the `recordings/` folder as `.csv` files. When recording stops, rows are resampled to a fixed 20 Hz timeline; numeric pose and sensor values are linearly interpolated between synchronized frames. Live sensor matching continues to use the original capture timestamps.

## Architecture

1. **BLE Manager Thread:** Concurrently manages asynchronous BLE connections to 5 devices, handles auto-reconnection, and pushes raw packets into a thread-safe queue.
2. **Computer Vision Thread:** Grabs frames from the camera, runs MediaPipe, extracts relevant bodily landmarks, pushes the annotated image to a queue, and pushes pose data to a thread-safe queue.
3. **Synchronization Engine Thread:** Consumes data from BLE and CV queues, aligning them using timestamps inside a sliding time window (50ms tolerance). Fuses aligned data into a complete "frame".
4. **Dataset Recorder:** Formats synchronized frames and writes them directly to disk.
5. **Dashboard UI:** The main process runs a non-blocking PyQt5 interface, consuming processed frames and status updates to present real-time visualization and controls.
