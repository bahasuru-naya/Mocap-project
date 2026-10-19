# Offline Recording Test & Accuracy Evaluation

`predict_pose_from_recording.py` tests saved data-collection CSV recordings by:
1. Validating CSV format and strict 20 Hz timestamp grid (50 ms per frame).
2. Generating 17 3-D pose landmark predictions (51 outputs) from 30 IMU channels using the ONNX ensemble model.
3. Automatically evaluating accuracy metrics when ground truth MediaPipe pose columns (`cv_0_x` ... `cv_16_z`) are present.
4. Exporting summary metrics across motions and per-joint error rankings to CSV.

## Quick Start (.bat file)

Double-click or run from command prompt:

```bat
# From repository root:
run_check_accuracy.bat

# Or from prediction_system:
cd prediction_system
run_check_accuracy.bat
```

By default, it evaluates all recording CSV files in `data_collection\recordings`.

You can also pass arguments to the batch file:

```bat
run_check_accuracy.bat "..\data_collection\recordings\Double Arm Swing.csv"
run_check_accuracy.bat --save-predictions
```

## Python CLI Usage

Run using `prediction_system\my_env`:

### 1. Evaluate All Recordings in Default Directory

```powershell
python prediction_system\Test\predict_pose_from_recording.py
```

### 2. Evaluate a Specific Recording File

```powershell
python prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\Double Arm Swing.csv"
```

### 3. Save Predictions CSV Alongside Recording

```powershell
python prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\Double Arm Swing.csv" --save-predictions
```

### 4. Custom Output Path for Predictions

```powershell
python prediction_system\Test\predict_pose_from_recording.py "data_collection\recordings\Double Arm Swing.csv" -o "output.csv"
```

### 5. Custom Summary Export Location

```powershell
python prediction_system\Test\predict_pose_from_recording.py --summary-csv "my_summary.csv"
```

## Metrics Computed

- **MPJPE (Mean Per-Joint Position Error)**: Euclidean distance in 3-D space between predicted and ground-truth keypoints, averaged across all frames and joints.
- **Per-Joint MPJPE**: Mean, median, and max error for each of the 17 skeleton keypoints (Head, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Heels, Foot Indices).
- **MAE & RMSE**: Coordinate-level Mean Absolute Error and Root Mean Squared Error (overall and decomposed into X, Y, Z axes).
- **PCK (Percentage of Correct Keypoints)**: Percentage of predicted joint positions within distance thresholds (e.g., PCK@0.05, PCK@0.10).
- **$R^2$ Score**: Variance-weighted and uniform average coefficient of determination.
- **Inference Speed**: Frames per second (FPS) and total inference runtime.
