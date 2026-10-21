# AI MoCap Data Collection System

A complete data collection system for synchronized AI motion capture, utilizing 5 BLE-enabled ESP32 + BMI160 nodes and a MediaPipe-based Computer Vision module.

## Setup Instructions

### Hardware Setup (Sensor Nodes)
1. You will need 5 ESP32 microcontrollers and 5 BMI160 sensors.
2. Wire each BMI160 sensor to its ESP32 using I2C.
3. Use the matching sketch in `IMUs/IMU_sensor_node_<n>/` for each board. All five sketches use the same firmware; only `NODE_ID` differs.
   The DFRobot BMI160 library configures the accelerometer to ±2 g and gyroscope to ±2000°/s. Firmware converts accelerometer counts at 16384 counts/g and maps the signed 16-bit gyro range (−32768..32767) to ±2000°/s before converting to rad/s.
4. Upload each matching sketch to its board:
   - `IMU_sensor_node_1/IMU_sensor_node_1.ino` -> `ESP32_IMU_1`
   - `IMU_sensor_node_2/IMU_sensor_node_2.ino` -> `ESP32_IMU_2`
   - `IMU_sensor_node_3/IMU_sensor_node_3.ino` -> `ESP32_IMU_3`
   - `IMU_sensor_node_4/IMU_sensor_node_4.ino` -> `ESP32_IMU_4`
   - `IMU_sensor_node_5/IMU_sensor_node_5.ino` -> `ESP32_IMU_5`
5. In Arduino IDE, select the correct board and port for each ESP32 and verify upload succeeds. Reflash all five boards after firmware/protocol changes so they all run the same version.

### PC Software Setup
1. Ensure you have Python 3.8+ installed.
2. On Windows 10/11, turn on Bluetooth in **Settings > Bluetooth & devices** (or the Bluetooth tile in Quick Settings). The PC must have a Bluetooth adapter that supports Bluetooth Low Energy (BLE); install or update its driver if Windows does not show a Bluetooth toggle. The application scans for the ESP32 nodes over BLE, so you do not need to pair them manually in Windows.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Connect a webcam to your PC.
5. Turn on your ESP32 modules and keep them within BLE range of the PC.

### Running the System
Run the main application:
```bash
python src/app_main.py
```

### Dashboard Usage
- **Calibration:** Before starting a recording, place all IMUs flat and stationary, and click "Calibrate All Sensors". Wait for the status indicator on each node to turn Blue ("CALIBRATED"). 
- **Recording:** Enter a session name and click "Start Recording".
- **Synchronization Monitor:** The saved dataset is aligned to a 20 Hz timeline (one row every 50 ms) after recording stops. "Live Readout Rate" reports dashboard telemetry updates only; it is not a synchronized data rate or the saved dataset sampling rate.
- **Packet-loss reporting:** The saved-dataset summary reports IMU sequence-number gaps separately from blank values remaining after 20 Hz resampling. The firmware increments its 16-bit sequence number once per transmitted packet. Gaps between received sequence numbers indicate missing notifications, but timer ticks coalesced while the firmware loop is busy are not reflected in the counter. Sequence numbers wrap, and gaps cannot identify a packet lost after the final received packet or determine every cause of a missing notification.

## Synchronization Details

The ESP32 nodes do not start from a shared hardware trigger. Each node samples its BMI160 on a 20 ms timer (nominally 50 Hz) and sends BLE notifications containing a per-node sequence number, its millisecond uptime, and the six accelerometer/gyroscope values. The PC also timestamps each notification when it arrives.

Before recording can start, the recorder collects at least 10 samples from every configured node. For each sample it subtracts the unwrapped ESP32 uptime from the PC's monotonic arrival time to get a candidate clock offset. It estimates each node's offset using the 5th percentile of up to 500 recent candidates, which reduces the effect of BLE delivery delay. This is a one-way arrival-time estimate: there is no two-way clock ping or round-trip delay measurement. The application uses the offsets to map each node's timestamps onto the PC monotonic timeline, which is also used to timestamp camera poses.

At recording start, the offsets are frozen. The application does not continually correct clock drift, so long recordings may accumulate timing error as independent device clocks drift. Raw IMU and camera samples are collected on the shared timeline; they are not aligned in real time. After recording stops, the recorder creates a uniform 20 Hz CSV timeline. IMU values are interpolated when neighboring samples are no more than 50 ms apart; at stream boundaries, the nearest IMU sample is used only when it is within 50 ms. Each camera row uses the nearest pose within 50 ms. Otherwise, IMU cells are left blank and missing pose landmarks are zero-padded. The final five seconds are intentionally excluded to reduce the chance of including unintended or random movement while the user stops the recording. The dashboard displays the latest raw IMU readouts and does not show the synchronized dataset rate.

## Dataset Format and Model Use

Each saved CSV row represents one 50 ms timeline slot. It contains:

- `sys_time` (Unix seconds) and `local_time` for the aligned slot.
- `cv_0_x` through `cv_16_z`: 17 MediaPipe world-pose landmarks (51 values) used as supervised pose targets. A missing camera pose is zero-padded.
- `ESP32_IMU_1_ax` through `ESP32_IMU_5_az`: accelerometer values in g after subtracting the stationary calibration offset, which includes gravity at the calibration orientation; these are not gravity-compensated linear accelerations. `ESP32_IMU_1_gx` through `ESP32_IMU_5_gz`: gyroscope angular rates in rad/s. Gyroscope packet fields use 0.01 rad/s units on the wire and are decoded to rad/s by the PC. A sensor value remains blank when no valid sample is available within the 50 ms synchronization limit.

Use the 30 IMU columns as model inputs and the 51 `cv_*` columns as pose labels; do not feed timestamp or `local_time` columns to the model. 

## Architecture

1. **BLE Manager Thread:** Concurrently manages asynchronous BLE connections to 5 devices, decodes fixed-size 20-byte IMU notifications containing sensor uptime, timestamps packet arrival using the PC monotonic clock, and pushes samples into a thread-safe queue.
2. **Computer Vision Thread:** Grabs frames from the camera, runs MediaPipe, extracts relevant bodily landmarks, pushes the annotated image to a queue, and pushes pose data to a thread-safe queue.
3. **Stream Forwarder:** Drains BLE and CV queues, forwards raw samples to the recorder, and shows latest IMU telemetry in the dashboard. It does not synchronize IMUs or align camera frames in realtime.
4. **Dataset Recorder:** Uses the shared clock-offset estimator to calibrate each ESP32-to-PC offset from pre-recording samples, freezes the offsets at recording start, logs raw IMU and camera samples on the PC monotonic timeline, and aligns them into a uniform 20 Hz CSV after recording stops. The saved CSV is the input to offline prediction and contains both IMU features and CV pose labels.
5. **Dashboard UI:** The main process runs a non-blocking PyQt5 interface, consuming the camera preview, raw latest-sample telemetry, and status updates to present visualization and controls.
