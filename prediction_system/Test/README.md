# Offline Recording Test

`predict_pose_from_recording.py` tests a saved data-collection CSV by
validating its format and 20 Hz timestamp grid, then generating pose
predictions from its IMU columns. It is an offline dataset test tool; it is
not part of real-time BLE prediction.

Run from the repository root:

```powershell
python prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\session.csv"
```

Or run it from `prediction_system`:

```powershell
python Test\predict_pose_from_recording.py "..\data_collection\recordings\session.csv"
```

By default, predictions are saved beside the input CSV as
`<session>_predictions.csv`. Use `--output` to select another output path.
