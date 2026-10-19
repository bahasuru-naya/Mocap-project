# Mocap-project

An end-to-end motion-capture project that collects synchronized data from five
BLE-connected ESP32/BMI160 IMUs and a webcam, then predicts 3-D body pose from
the IMU streams. It includes a data-collection application, a real-time
prediction application, pre-trained model artifacts, and an offline tool for
generating predictions from recorded CSV files.

## System overview

```text
Five ESP32/BMI160 nodes ──BLE──┬── Data collection app ── synchronized CSV
                              └── Live prediction app ── 3-D pose dashboard
Webcam ── MediaPipe ───────────── Data collection app ── pose labels in CSV
Recorded CSV ──────────────────── Offline prediction tool ── prediction CSV
```

Each node sends accelerometer and gyroscope measurements, a sequence number,
and its uptime over BLE. The collection application also captures camera pose
landmarks using MediaPipe. Collection and live prediction both use the shared
clock-offset estimator and align IMU samples to a 20 Hz timeline.

## Hardware

- Five ESP32 boards, each connected over I2C to a BMI160 sensor.
- A PC with Bluetooth Low Energy support.
- A webcam for data collection; live IMU prediction does not use camera data.

Use the matching sketch in
[`data_collection/IMUs/`](./data_collection/IMUs/) for each node. The sketches
share the same firmware protocol and differ by node ID. The BMI160
accelerometer range is ±2 g and the gyroscope range is ±2000°/s. The firmware
converts accelerometer readings to g and gyroscope readings to rad/s; see the
[collection guide](./data_collection/README_MOCAP.md) for details. Reflash all
five boards after firmware or protocol changes.

## Install

Use separate Python environments for the two applications because they have
different dependency sets. From the repository root, create and activate one
environment per application, then install its requirements:

```powershell
python -m venv data_collection\.venv
data_collection\.venv\Scripts\Activate.ps1
python -m pip install -r data_collection\requirements.txt
```

For prediction, open a separate PowerShell session:

```powershell
python -m venv prediction_system\my_env
prediction_system\my_env\Scripts\Activate.ps1
python -m pip install -r prediction_system\requirements.txt
```

On Windows, enable Bluetooth in **Settings > Bluetooth & devices** and ensure
the adapter supports BLE. The applications scan for the ESP32 nodes; manual
Windows pairing is not required.

## Run the applications

### Collect a dataset

Connect the webcam, power on the five sensor nodes, and run:

```powershell
data_collection\.venv\Scripts\python.exe data_collection\src\app_main.py
```

In the dashboard, calibrate the sensors while they are flat and stationary,
wait for their calibrated status, enter a session name, then select **Start
Recording**. Select **Stop Recording** when finished. CSV recordings are saved
under `data_collection\recordings\`.

Before recording starts, the application gathers at least 10 clock
observations from each IMU. It estimates each offset from PC monotonic arrival
time minus unwrapped node uptime, using the 5th percentile of up to 500 recent
observations, and freezes the estimates for the recording. After recording,
the raw streams are aligned to a 20 Hz (50 ms) CSV timeline. Samples outside
the 50 ms synchronization tolerance are left missing (camera pose labels are
zero-padded when unavailable), and the final five seconds are excluded.
Offsets are not continuously corrected for clock drift.

The CSV contains 30 IMU input features (five nodes × six axes) and 51 camera
pose values (17 landmarks × x/y/z). Accelerometer values are in g after
stationary offset calibration, including gravity at the calibration
orientation; they are not gravity-compensated linear acceleration. Gyroscope
values are in rad/s. Use the IMU columns as model inputs and `cv_*` columns as
pose targets; exclude timestamp columns.

For setup, dashboard controls, synchronization details, and the full CSV
schema, see the [data-collection guide](./data_collection/README_MOCAP.md).

### Predict pose live

Run:

```powershell
prediction_system\my_env\Scripts\python.exe prediction_system\src\app_main.py
```

The application streams the five IMUs and displays the ensemble's estimated
17-landmark 3-D pose. Once clock synchronization has at least 10 observations
for every node, choose **Start Prediction**. **Stop Prediction** halts
inference and clears its rolling input window; BLE streaming and clock
synchronization continue. Live prediction uses no webcam input.

The live synchronizer freezes the same shared clock-offset estimates used by
collection, then interpolates or holds samples on a 20 Hz grid using a 50 ms
tolerance. The model takes a 10-row window of 30 IMU features. Three ONNX
ensemble members are averaged, and the pose scaler converts the result back
to landmark coordinates. Required ONNX models and scalers are in
`prediction_system\model\` and `prediction_system\scalers\`.

More information is available in the
[prediction-system guide](./prediction_system/README.md).

### Predict from a recorded CSV

The offline tool uses the same model, scalers, and inference pipeline as live
prediction. From the repository root:

```powershell
prediction_system\my_env\Scripts\python.exe prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\session.csv"
```

It writes `<session>_predictions.csv` alongside the input by default, retaining
the original columns and adding 51 `pred_cv_*` columns. The first nine input
rows have no prediction because a full 10-row input window is required. Use
`--output` to specify another output path. See the
[prediction guide](./prediction_system/README.md) for validation rules and
additional usage details.

## Project contents

- [`data_collection/`](./data_collection/) — ESP32 firmware, BLE and camera
  collection application, synchronized CSV recorder, and setup guide.
- [`prediction_system/`](./prediction_system/) — live and offline prediction
  applications, model/scaler artifacts, and usage guide.
- [`common/`](./common/) — shared clock synchronization implementation and
  tests.
- [`model training/pose-from-imu.ipynb`](./model%20training/pose-from-imu.ipynb)
  — model-training notebook.

The checked-in model artifacts can be used for inference. The repository does
not include a complete, reproducible training procedure for regenerating those
artifacts; see the prediction guide for details.
