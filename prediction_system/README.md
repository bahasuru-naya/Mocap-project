# Motion Prediction System

This application maps the five synchronized IMU streams to 17 three-dimensional
pose landmarks. It supports live BLE inference and offline inference from CSV
files produced by [data collection](../data_collection/README_MOCAP.md).

## Requirements and models

Install the packages listed in `requirements.txt` in a Python environment. The
system expects these files:

- `model/best_model_overall_member0.onnx` through `member2.onnx`
- `scalers/imu_scaler.joblib` and `scalers/pose_scaler.joblib`

The checked-in ensemble accepts a 10-row window of 30 ordered IMU features and
returns 51 scaled pose values. The shared inference pipeline applies the IMU
scaler, averages the three ONNX model outputs, reverses pose scaling, and
reshapes the result into 17 `(x, y, z)` landmarks.

For a single-window prediction, `PoseEnsemble.predict_pose_from_imu` accepts a
pandas DataFrame with the 30 IMU columns and 10 rows in oldest-to-newest order.
It selects the columns in the training order, runs the same scaler/ensemble/
inverse-scaler steps, and returns a pandas Series indexed by the 51 `cv_*`
pose columns in original units. Both live inference and the recorded-CSV tool
use this same window prediction path.

## Test a recorded dataset (offline inference)

The collector saves synchronized CSVs at 20 Hz (one row every 50 ms). This
offline test script reads a saved recording and generates pose predictions; it
is not used by the real-time BLE application. Run it from the repository root
or from `prediction_system`:

```powershell
python prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\session.csv"
```

Or, from `prediction_system`:

```powershell
python Test\predict_pose_from_recording.py "..\data_collection\recordings\session.csv"
```

The script validates the required `sys_time` and 30 IMU columns, checks the
20 Hz timestamp grid, and uses the same ensemble, scalers, and missing-value
interpolation as live inference. Each output prediction uses the current row
and the preceding nine rows; therefore the first nine input rows have no
prediction. The output is written beside the input as
`<session>_predictions.csv` (or to the path supplied with `--output`). It keeps
the original columns, including `cv_*` reference landmarks, and adds 51
`pred_cv_*` columns for comparison. These are pose estimates, not a future-time
forecast.

CSV rows with blank IMU values are accepted and gaps are interpolated within
each model window. A feature that is entirely missing within a window is
filled with zero, matching the real-time prediction pipeline. Files with
non-monotonic or irregular timestamps are rejected rather than silently
resampled; collect data with the data-collection application to produce the
expected 20 Hz timeline. Camera pose columns are not required for inference,
but provide reference labels for later evaluation or training.

## Live BLE prediction

Run the Qt dashboard to stream the IMUs and display the ensemble's current
pose estimate:

```powershell
python prediction_system\src\app_main.py
```

Before inference begins, live mode collects at least 10 BLE-arrival clock
observations from each IMU. It uses the same shared estimator as data
collection: unwrap each ESP32 `millis()` timestamp, calculate PC-monotonic
arrival-minus-uptime offsets, retain up to 500 observations, and choose the
5th-percentile offset. It freezes those offsets and maps all devices onto PC
monotonic time, then interpolates/holds samples onto a 20 Hz (50 ms) grid.
Interpolation is limited to adjacent samples no more than 50 ms apart, and a
nearest sample is held only when it is within 50 ms of the target time. This
matches the data collector's live IMU synchronization tolerance; live
prediction does not use camera data. Offsets are not updated during the
session, so clock drift is not corrected. Use **Start Prediction** once clock
synchronization is ready to begin live inference; **Stop Prediction** halts
inference and clears its rolling input window while BLE streaming and clock
synchronization continue. Live mode does not read a saved dataset. For
post-recording inference against the saved CSV, use
`Test\predict_pose_from_recording.py`.

## Training data

The collection CSV contains 30 IMU input features and 51 camera-derived pose
targets per aligned row. Use the IMU columns as `X` and the matching `cv_*`
columns as `y`; exclude `sys_time` and `local_time`. Filter or mask rows with
zero-padded missing poses, and handle blank sensor cells deliberately. The
repository includes pre-trained model/scaler artifacts and conversion code,
but does not currently provide the training procedure that generated those
artifacts. Retraining requires a compatible supervised training pipeline and
regenerating both scalers and all ensemble members together.
