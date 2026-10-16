# AI MoCap Data Collection System

A complete data collection system for synchronized AI motion capture, utilizing 5 BLE-enabled ESP32 + BMI160 nodes and a MediaPipe-based Computer Vision module.

## Setup Instructions

### Hardware Setup (Sensor Nodes)
1. You will need 5 ESP32 microcontrollers and 5 BMI160 sensors.
2. Wire each BMI160 sensor to its ESP32 using I2C.
3. Use the matching sketch in `IMUs/IMU_sensor_node_<n>/` for each board. All five sketches use the same firmware; only `NODE_ID` differs.
4. Upload each matching sketch to its board:
   - `IMU_sensor_node_1/IMU_sensor_node_1.ino` -> `ESP32_IMU_1`
   - `IMU_sensor_node_2/IMU_sensor_node_2.ino` -> `ESP32_IMU_2`
   - `IMU_sensor_node_3/IMU_sensor_node_3.ino` -> `ESP32_IMU_3`
   - `IMU_sensor_node_4/IMU_sensor_node_4.ino` -> `ESP32_IMU_4`
   - `IMU_sensor_node_5/IMU_sensor_node_5.ino` -> `ESP32_IMU_5`
5. In Arduino IDE, select the correct board and port for each ESP32 and verify upload succeeds. Reflash all five boards after firmware/protocol changes so they all run the same version.

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
- **Synchronization Monitor:** The saved dataset is aligned to a 20 Hz timeline (one row every 50 ms) after recording stops. "Live Readout Rate" reports dashboard telemetry updates only; it is not a synchronized data rate or the saved dataset sampling rate.
- **Data Output:** Before recording can start, the recorder observes at least 10 IMU samples from every configured node and estimates each ESP32 uptime offset against the PC monotonic clock from BLE sample-arrival observations. It selects the 5th percentile of up to 500 offsets to reduce BLE-delay bias. This is a one-way arrival-time estimate, not a two-way clock-ping measurement or low-round-trip sample selection. The same offset estimator and timestamp-unwrapping logic are used for live prediction. Offsets are frozen when recording starts; the system does not check or correct clock drift during a recording. Raw IMU and camera samples are logged against the shared PC-monotonic timeline, without realtime IMU/CV alignment. After recording stops, the saved dataset is resampled onto a uniform 20 Hz timeline. IMU values are interpolated between samples no more than 150 ms apart; the nearest sample is used at stream boundaries if within 150 ms. The nearest camera pose within 150 ms is used. The final 5 seconds are intentionally excluded. Missing measurements or gaps longer than 150 ms remain blank (or zero-padded for CV landmarks). The live dashboard only displays the latest raw IMU readout; its update rate is not the dataset rate.
- **Packet-loss reporting:** The saved-dataset summary reports IMU sequence-number gaps separately from blank values remaining after 20 Hz resampling. The firmware advances its sequence by elapsed timer ticks, so timer ticks missed while its loop is busy also appear as gaps. Sequence gaps expose losses between received packets; they cannot identify a packet lost after the final received packet or distinguish every cause of a gap.

## Dataset Format and Model Use

Each saved CSV row represents one 50 ms timeline slot. It contains:

- `sys_time` (Unix seconds) and `local_time` for the aligned slot.
- `cv_0_x` through `cv_16_z`: 17 MediaPipe world-pose landmarks (51 values) used as supervised pose targets. A missing camera pose is zero-padded.
- `ESP32_IMU_1_ax` through `ESP32_IMU_5_gz`: six accelerometer/gyroscope values per IMU (30 values). A sensor value remains blank when no valid sample is available within the 150 ms synchronization limit.

Use the 30 IMU columns as model inputs and the 51 `cv_*` columns as pose labels; do not feed timestamp or `local_time` columns to the model. Rows are synchronized but not necessarily complete: training should account for blank IMU cells and zero-padded missing poses rather than treating them as measured values. The last five seconds of each recording are intentionally omitted. See [the prediction-system guide](../prediction_system/README.md) for the model input contract and offline inference command. The repository includes inference artifacts; it does not currently include a model-training script.

## Architecture

1. **BLE Manager Thread:** Concurrently manages asynchronous BLE connections to 5 devices, decodes fixed-size 20-byte IMU notifications containing sensor uptime, timestamps packet arrival using the PC monotonic clock, and pushes samples into a thread-safe queue.
2. **Computer Vision Thread:** Grabs frames from the camera, runs MediaPipe, extracts relevant bodily landmarks, pushes the annotated image to a queue, and pushes pose data to a thread-safe queue.
3. **Stream Forwarder:** Drains BLE and CV queues, forwards raw samples to the recorder, and shows latest IMU telemetry in the dashboard. It does not synchronize IMUs or align camera frames in realtime.
4. **Dataset Recorder:** Uses the shared clock-offset estimator to calibrate each ESP32-to-PC offset from pre-recording samples, freezes the offsets at recording start, logs raw IMU and camera samples on the PC monotonic timeline, and aligns them into a uniform 20 Hz CSV after recording stops. The saved CSV is the input to offline prediction and contains both IMU features and CV pose labels.
5. **Dashboard UI:** The main process runs a non-blocking PyQt5 interface, consuming the camera preview, raw latest-sample telemetry, and status updates to present visualization and controls.
