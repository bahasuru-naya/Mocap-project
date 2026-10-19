"""Generate pose predictions and evaluate model accuracy from recorded data-collection CSVs."""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREDICTION_ROOT / "src"))

from system.prediction_engine import (
    IMU_COLS,
    POSE_COLS,
    PoseEnsemble,
)

SAMPLE_PERIOD_SECONDS = 0.05
SAMPLE_PERIOD_TOLERANCE_SECONDS = 0.002

KEYPOINT_NAMES = [
    "Head",              # 0
    "Left Shoulder",     # 1
    "Right Shoulder",    # 2
    "Left Elbow",        # 3
    "Right Elbow",       # 4
    "Left Wrist",        # 5
    "Right Wrist",       # 6
    "Left Hip",          # 7
    "Right Hip",         # 8
    "Left Knee",         # 9
    "Right Knee",        # 10
    "Left Ankle",        # 11
    "Right Ankle",       # 12
    "Left Heel",         # 13
    "Right Heel",        # 14
    "Left Foot Index",   # 15
    "Right Foot Index",  # 16
]


def _read_dataset(path):
    """Read and validate a recorded CSV file.

    Returns:
        columns: list of column names in the CSV.
        rows: list of original row dictionaries.
        feature_array: (N, 30) float32 numpy array of IMU readings.
        missing_values: count of missing IMU entries.
        gt_pose_array: (N, 17, 3) float32 numpy array if POSE_COLS exist, else None.
    """
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("CSV is empty or has no header row.")
        if len(columns) != len(set(columns)):
            raise ValueError("CSV has duplicate column names.")

        required_columns = {"sys_time", *IMU_COLS}
        missing_columns = sorted(required_columns.difference(columns))
        if missing_columns:
            raise ValueError(
                "CSV is missing required collection columns: "
                + ", ".join(missing_columns)
            )

        has_ground_truth = all(col in columns for col in POSE_COLS)

        rows = []
        timestamps = []
        feature_rows = []
        gt_pose_rows = []
        missing_values = 0

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["sys_time"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid sys_time on CSV line {line_number}.") from error
            if not math.isfinite(timestamp):
                raise ValueError(f"Non-finite sys_time on CSV line {line_number}.")

            features = []
            for column in IMU_COLS:
                value = row.get(column)
                if value is None or not value.strip():
                    features.append(np.nan)
                    missing_values += 1
                    continue
                try:
                    numeric_value = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid numeric value in {column} on CSV line {line_number}."
                    ) from error
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Non-finite value in {column} on CSV line {line_number}."
                    )
                features.append(numeric_value)

            if has_ground_truth:
                gt_coords = []
                for column in POSE_COLS:
                    val = row.get(column)
                    try:
                        num = float(val) if (val is not None and val.strip()) else np.nan
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid numeric ground truth in {column} on line {line_number}."
                        ) from error
                    gt_coords.append(num)
                gt_pose_rows.append(gt_coords)

            rows.append(row)
            timestamps.append(timestamp)
            feature_rows.append(features)

    if not rows:
        raise ValueError("CSV has no data rows.")

    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError("sys_time must increase strictly in collection order.")
    if not np.all(
        np.abs(intervals - SAMPLE_PERIOD_SECONDS) <= SAMPLE_PERIOD_TOLERANCE_SECONDS
    ):
        median_interval = float(np.median(intervals))
        raise ValueError(
            "The model expects the collection system's regular 20 Hz grid "
            f"(50 ms per row); observed median interval {median_interval:.6f} s."
        )

    gt_pose_arr = None
    if has_ground_truth and gt_pose_rows:
        gt_pose_arr = np.asarray(gt_pose_rows, dtype=np.float32).reshape(-1, 17, 3)

    return (
        columns,
        rows,
        np.asarray(feature_rows, dtype=np.float32),
        missing_values,
        gt_pose_arr,
    )


def compute_accuracy_metrics(y_true, y_pred, pck_thresholds=(0.05, 0.10)):
    """Calculate comprehensive accuracy metrics between ground-truth and predicted poses.

    Args:
        y_true: (N, 17, 3) array of ground-truth 3-D landmarks.
        y_pred: (N, 17, 3) array of predicted 3-D landmarks.
        pck_thresholds: tuple of distance thresholds for Percentage of Correct Keypoints.

    Returns:
        dict of computed metrics.
    """
    diff = y_pred - y_true  # (N, 17, 3)
    dist = np.linalg.norm(diff, axis=-1)  # (N, 17) Euclidean distance per joint

    mpjpe = float(np.mean(dist))
    per_joint_mpjpe = np.mean(dist, axis=0)  # (17,)
    per_joint_median = np.median(dist, axis=0)
    per_joint_max = np.max(dist, axis=0)

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Per-axis (X, Y, Z) metrics
    axis_mae = np.mean(np.abs(diff), axis=(0, 1))  # (3,)
    axis_rmse = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))  # (3,)

    pck = {th: float(np.mean(dist < th) * 100.0) for th in pck_thresholds}

    n_samples = len(y_true)
    flat_true = y_true.reshape(n_samples, 51)
    flat_pred = y_pred.reshape(n_samples, 51)
    r2_vw = float(r2_score(flat_true, flat_pred, multioutput="variance_weighted"))
    r2_uni = float(r2_score(flat_true, flat_pred, multioutput="uniform_average"))

    return {
        "samples": n_samples,
        "mpjpe": mpjpe,
        "median_err": float(np.median(dist)),
        "std_err": float(np.std(dist)),
        "max_err": float(np.max(dist)),
        "per_joint_mpjpe": per_joint_mpjpe,
        "per_joint_median": per_joint_median,
        "per_joint_max": per_joint_max,
        "mae": mae,
        "rmse": rmse,
        "axis_mae": axis_mae,
        "axis_rmse": axis_rmse,
        "pck": pck,
        "r2_variance_weighted": r2_vw,
        "r2_uniform": r2_uni,
    }


def evaluate_dataset(
    dataset_path,
    predictor,
    output_path=None,
    save_predictions=False,
    pck_thresholds=(0.05, 0.10),
):
    """Run model inference and evaluate accuracy for a single recording CSV."""
    columns, rows, features, missing_values, gt_poses = _read_dataset(dataset_path)

    window_size = predictor.window_size
    n_rows = len(rows)
    if n_rows < window_size:
        raise ValueError(
            f"Dataset has {n_rows} rows; the loaded model requires "
            f"{window_size} rows per window."
        )

    n_predictions = n_rows - window_size + 1

    # Run inference across windows using direct numpy slices for speed
    t0 = time.perf_counter()
    predictions = np.empty((n_predictions, 17, 3), dtype=np.float32)
    for idx, end_index in enumerate(range(window_size - 1, n_rows)):
        window = features[end_index - window_size + 1 : end_index + 1]
        predictions[idx] = predictor.predict_window(window)
    infer_time = time.perf_counter() - t0
    fps = n_predictions / infer_time if infer_time > 0 else 0.0

    # Save predictions CSV if requested
    if save_predictions:
        target_csv = (
            output_path
            if output_path
            else dataset_path.with_name(f"{dataset_path.stem}_predictions.csv")
        )
        if target_csv == dataset_path:
            raise ValueError("The output path must not overwrite the input dataset.")

        pred_flat = predictions.reshape(n_predictions, 51)
        pred_cols = [f"pred_{col}" for col in POSE_COLS]
        with target_csv.open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(
                dest,
                fieldnames=[*columns, *pred_cols],
                extrasaction="ignore",
            )
            writer.writeheader()
            for idx, end_index in enumerate(range(window_size - 1, n_rows)):
                out_row = dict(rows[end_index])
                for col_name, val in zip(pred_cols, pred_flat[idx]):
                    out_row[col_name] = repr(float(val))
                writer.writerow(out_row)
        print(f"  [Output] Predictions saved to: {target_csv}")

    metrics = None
    if gt_poses is not None:
        gt_target = gt_poses[window_size - 1 : n_rows]
        metrics = compute_accuracy_metrics(gt_target, predictions, pck_thresholds=pck_thresholds)
        metrics["dataset_name"] = dataset_path.stem
        metrics["file_path"] = str(dataset_path)
        metrics["infer_time"] = infer_time
        metrics["fps"] = fps
        metrics["missing_imu_values"] = missing_values

    return {
        "dataset_name": dataset_path.stem,
        "file_path": str(dataset_path),
        "total_rows": n_rows,
        "predicted_rows": n_predictions,
        "missing_imu_values": missing_values,
        "infer_time": infer_time,
        "fps": fps,
        "has_ground_truth": gt_poses is not None,
        "metrics": metrics,
        "predictions": predictions,
        "gt_poses": gt_poses[window_size - 1 : n_rows] if gt_poses is not None else None,
    }


def print_single_report(res, pck_thresholds=(0.05, 0.10)):
    """Print a clean accuracy and inference report for a single dataset."""
    m = res["metrics"]
    print("=" * 80)
    print(f" RECORDING: {res['dataset_name']}")
    print("=" * 80)
    print(f"  File:              {res['file_path']}")
    print(f"  Frames Processed:  {res['predicted_rows']} (Total Rows: {res['total_rows']})")
    print(f"  Missing IMU Vals:  {res['missing_imu_values']}")
    print(f"  Inference Speed:   {res['fps']:.1f} FPS ({res['infer_time']:.2f} s total)")
    print("-" * 80)

    if not res["has_ground_truth"] or m is None:
        print("  [Notice] No ground truth 'cv_*' columns present in this CSV.")
        print("  Pose predictions were generated successfully, but accuracy cannot be computed.")
        print("=" * 80 + "\n")
        return

    pck_str = " | ".join([f"PCK@{th:.2f}: {m['pck'][th]:5.2f}%" for th in pck_thresholds])
    print(f"  MPJPE (Mean):      {m['mpjpe']:.4f} m (or norm units)")
    print(f"  Median Joint Err:  {m['median_err']:.4f}")
    print(f"  Std Dev Joint Err: {m['std_err']:.4f}")
    print(f"  Max Joint Err:     {m['max_err']:.4f}")
    print(f"  MAE (Coordinate):  {m['mae']:.4f} [X: {m['axis_mae'][0]:.4f}, Y: {m['axis_mae'][1]:.4f}, Z: {m['axis_mae'][2]:.4f}]")
    print(f"  RMSE (Coordinate): {m['rmse']:.4f} [X: {m['axis_rmse'][0]:.4f}, Y: {m['axis_rmse'][1]:.4f}, Z: {m['axis_rmse'][2]:.4f}]")
    print(f"  Accuracy (PCK):    {pck_str}")
    print(f"  R2 Score (vw):     {m['r2_variance_weighted']:.4f}  |  R2 (uniform): {m['r2_uniform']:.4f}")
    print("-" * 80)
    print("  PER-JOINT POSITION ERROR BREAKDOWN (MPJPE):")
    print(f"  {'ID':>3}  {'Keypoint':<18} {'MPJPE (m)':>10} {'Median':>10} {'Max':>10}")
    print("  " + "-" * 56)
    for j_idx, name in enumerate(KEYPOINT_NAMES):
        print(
            f"  {j_idx:>3}  {name:<18} "
            f"{m['per_joint_mpjpe'][j_idx]:>10.4f} "
            f"{m['per_joint_median'][j_idx]:>10.4f} "
            f"{m['per_joint_max'][j_idx]:>10.4f}"
        )
    print("=" * 80 + "\n")


def print_overall_summary(results, pck_thresholds=(0.05, 0.10)):
    """Print consolidated comparison table and joint error ranking across all recordings."""
    valid_results = [r for r in results if r["metrics"] is not None]
    if not valid_results:
        return

    print("\n" + "=" * 102)
    print("                            PREDICTION MODEL ACCURACY SUMMARY")
    print("=" * 102)
    th1, th2 = pck_thresholds[0], pck_thresholds[1]
    header = (
        f"{'Recording':<28} "
        f"{'Frames':>7} "
        f"{'MPJPE':>8} "
        f"{'MAE':>8} "
        f"{'RMSE':>8} "
        f"{f'PCK@{th1:.2f}':>9} "
        f"{f'PCK@{th2:.2f}':>9} "
        f"{'R2 (vw)':>8} "
        f"{'FPS':>7}"
    )
    print(header)
    print("-" * 102)

    all_gt = []
    all_preds = []
    total_frames = 0
    total_infer_time = 0.0

    for r in valid_results:
        m = r["metrics"]
        total_frames += r["predicted_rows"]
        total_infer_time += r["infer_time"]
        if r["gt_poses"] is not None and r["predictions"] is not None:
            all_gt.append(r["gt_poses"])
            all_preds.append(r["predictions"])

        line = (
            f"{r['dataset_name']:<28} "
            f"{r['predicted_rows']:>7d} "
            f"{m['mpjpe']:>8.4f} "
            f"{m['mae']:>8.4f} "
            f"{m['rmse']:>8.4f} "
            f"{m['pck'][th1]:>8.2f}% "
            f"{m['pck'][th2]:>8.2f}% "
            f"{m['r2_variance_weighted']:>8.4f} "
            f"{r['fps']:>7.1f}"
        )
        print(line)

    print("-" * 102)

    # Calculate overall pooled metrics across all frames
    if all_gt and all_preds:
        pooled_gt = np.concatenate(all_gt, axis=0)
        pooled_pred = np.concatenate(all_preds, axis=0)
        pooled_metrics = compute_accuracy_metrics(pooled_gt, pooled_pred, pck_thresholds=pck_thresholds)
        overall_fps = total_frames / total_infer_time if total_infer_time > 0 else 0.0

        overall_line = (
            f"{'OVERALL (POOLED)':<28} "
            f"{total_frames:>7d} "
            f"{pooled_metrics['mpjpe']:>8.4f} "
            f"{pooled_metrics['mae']:>8.4f} "
            f"{pooled_metrics['rmse']:>8.4f} "
            f"{pooled_metrics['pck'][th1]:>8.2f}% "
            f"{pooled_metrics['pck'][th2]:>8.2f}% "
            f"{pooled_metrics['r2_variance_weighted']:>8.4f} "
            f"{overall_fps:>7.1f}"
        )
        print(overall_line)
        print("=" * 102)

        # Print consolidated per-joint ranking
        print("\n" + "-" * 72)
        print("  CONSOLIDATED PER-JOINT MPJPE RANKING (ACROSS ALL RECORDINGS)")
        print("-" * 72)
        print(f"  {'Rank':>4}  {'ID':>3}  {'Keypoint':<18} {'Mean MPJPE':>12} {'Median':>12} {'Max':>10}")
        print("  " + "-" * 68)

        # Sort joints by lowest error (best predicted) to highest
        joint_order = np.argsort(pooled_metrics["per_joint_mpjpe"])
        for rank, j_idx in enumerate(joint_order, start=1):
            name = KEYPOINT_NAMES[j_idx]
            print(
                f"  {rank:>4}  {j_idx:>3}  {name:<18} "
                f"{pooled_metrics['per_joint_mpjpe'][j_idx]:>12.4f} "
                f"{pooled_metrics['per_joint_median'][j_idx]:>12.4f} "
                f"{pooled_metrics['per_joint_max'][j_idx]:>10.4f}"
            )
        print("-" * 72 + "\n")

        return pooled_metrics

    return None


def export_summary_csv(results, pooled_metrics, output_csv_path, pck_thresholds=(0.05, 0.10)):
    """Export summary metrics table to CSV file."""
    th1, th2 = pck_thresholds[0], pck_thresholds[1]
    fieldnames = [
        "Recording",
        "Frames",
        "Missing_IMU_Values",
        "MPJPE",
        "Median_Err",
        "MAE",
        "RMSE",
        f"PCK_{th1:.2f}",
        f"PCK_{th2:.2f}",
        "R2_Variance_Weighted",
        "R2_Uniform",
        "FPS",
    ]
    # Add per-joint MPJPE columns
    for name in KEYPOINT_NAMES:
        fieldnames.append(f"MPJPE_{name.replace(' ', '_')}")

    rows = []
    for r in results:
        m = r["metrics"]
        if m is None:
            continue
        row_data = {
            "Recording": r["dataset_name"],
            "Frames": r["predicted_rows"],
            "Missing_IMU_Values": r["missing_imu_values"],
            "MPJPE": f"{m['mpjpe']:.6f}",
            "Median_Err": f"{m['median_err']:.6f}",
            "MAE": f"{m['mae']:.6f}",
            "RMSE": f"{m['rmse']:.6f}",
            f"PCK_{th1:.2f}": f"{m['pck'][th1]:.2f}",
            f"PCK_{th2:.2f}": f"{m['pck'][th2]:.2f}",
            "R2_Variance_Weighted": f"{m['r2_variance_weighted']:.6f}",
            "R2_Uniform": f"{m['r2_uniform']:.6f}",
            "FPS": f"{r['fps']:.1f}",
        }
        for j_idx, name in enumerate(KEYPOINT_NAMES):
            row_data[f"MPJPE_{name.replace(' ', '_')}"] = f"{m['per_joint_mpjpe'][j_idx]:.6f}"
        rows.append(row_data)

    if pooled_metrics:
        total_frames = sum(r["predicted_rows"] for r in results if r["metrics"] is not None)
        total_missing = sum(r["missing_imu_values"] for r in results if r["metrics"] is not None)
        total_infer_time = sum(r["infer_time"] for r in results if r["metrics"] is not None)
        overall_fps = total_frames / total_infer_time if total_infer_time > 0 else 0.0

        overall_row = {
            "Recording": "OVERALL_POOLED",
            "Frames": total_frames,
            "Missing_IMU_Values": total_missing,
            "MPJPE": f"{pooled_metrics['mpjpe']:.6f}",
            "Median_Err": f"{pooled_metrics['median_err']:.6f}",
            "MAE": f"{pooled_metrics['mae']:.6f}",
            "RMSE": f"{pooled_metrics['rmse']:.6f}",
            f"PCK_{th1:.2f}": f"{pooled_metrics['pck'][th1]:.2f}",
            f"PCK_{th2:.2f}": f"{pooled_metrics['pck'][th2]:.2f}",
            "R2_Variance_Weighted": f"{pooled_metrics['r2_variance_weighted']:.6f}",
            "R2_Uniform": f"{pooled_metrics['r2_uniform']:.6f}",
            "FPS": f"{overall_fps:.1f}",
        }
        for j_idx, name in enumerate(KEYPOINT_NAMES):
            overall_row[f"MPJPE_{name.replace(' ', '_')}"] = f"{pooled_metrics['per_joint_mpjpe'][j_idx]:.6f}"
        rows.append(overall_row)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary metrics exported to: {output_csv_path}")


def main():
    default_recordings_dir = (PREDICTION_ROOT.parent / "data_collection" / "recordings").resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate accuracy and/or predict 17 3-D pose landmarks from 20 Hz "
            "data-collection recordings using the trained ONNX ensemble model."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(default_recordings_dir),
        help=(
            "Path to a recorded CSV file OR a directory containing recorded CSV files. "
            f"(default: {default_recordings_dir})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for output predictions CSV (only valid when evaluating a single dataset).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions CSV alongside dataset(s). (default: False to conserve disk space).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to export accuracy summary comparison as CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PREDICTION_ROOT / "model",
        help="Directory containing ONNX model files (default: prediction_system/model).",
    )
    parser.add_argument(
        "--scalers-dir",
        type=Path,
        default=PREDICTION_ROOT / "scalers",
        help="Directory containing scaler ONNX files (default: prediction_system/scalers).",
    )
    parser.add_argument(
        "--prefer-gpu",
        action="store_true",
        help="Prefer GPU acceleration (DirectML / CUDA) if available.",
    )

    args = parser.parse_args()

    input_path = Path(args.dataset).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    # Collect list of CSV files to evaluate
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            parser.error(f"Input file must be a .csv file: {input_path}")
        csv_files = [input_path]
        single_mode = True
    elif input_path.is_dir():
        # Exclude predictions, metrics or summary files
        csv_files = sorted(
            [
                p
                for p in input_path.glob("*.csv")
                if not (
                    p.stem.endswith("_predictions")
                    or p.stem.endswith("_summary")
                    or p.stem.endswith("_metrics")
                )
            ]
        )
        if not csv_files:
            parser.error(f"No recording CSV files found in directory: {input_path}")
        single_mode = False
        if args.output and not args.output.is_dir():
            parser.error(
                "When input is a directory, --output must either be omitted or be an existing directory."
            )
    else:
        parser.error(f"Invalid input path: {input_path}")

    # Validate ONNX model and scaler files
    model_paths = [
        str(args.models_dir / f"best_model_overall_member{i}.onnx")
        for i in range(3)
    ]
    imu_scaler_path = str(args.scalers_dir / "imu_scaler.onnx")
    pose_scaler_path = str(args.scalers_dir / "pose_scaler.onnx")

    missing_files = [
        p for p in (*model_paths, imu_scaler_path, pose_scaler_path)
        if not os.path.isfile(p)
    ]
    if missing_files:
        parser.error(
            "Missing ONNX file(s):\n  " + "\n  ".join(missing_files)
            + "\nRun src/convert_model_to_onnx.py first to generate ONNX files."
        )

    # Initialize predictor
    print("\n[Accuracy Check] Initializing PoseEnsemble predictor...")
    try:
        predictor = PoseEnsemble(
            model_paths,
            imu_scaler_path,
            pose_scaler_path,
            prefer_gpu=args.prefer_gpu,
        )
    except Exception as e:
        parser.error(f"Failed to initialize PoseEnsemble: {e}")

    print(f"[Accuracy Check] Found {len(csv_files)} recording file(s) to evaluate.")
    print(f"[Accuracy Check] Model window length: {predictor.window_size} frames (50 ms / 20 Hz)\n")

    results = []
    pck_thresholds = (0.05, 0.10)

    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing: {csv_file.name} ...")
        out_file = None
        if single_mode and args.output:
            out_file = args.output.expanduser().resolve()
        elif not single_mode and args.output:
            out_file = args.output.expanduser().resolve() / f"{csv_file.stem}_predictions.csv"

        try:
            res = evaluate_dataset(
                dataset_path=csv_file,
                predictor=predictor,
                output_path=out_file,
                save_predictions=args.save_predictions or (single_mode and bool(args.output)),
                pck_thresholds=pck_thresholds,
            )
            results.append(res)
            if single_mode:
                print_single_report(res, pck_thresholds=pck_thresholds)
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {csv_file.name}: {e}")

    pooled_metrics = None
    if not single_mode:
        pooled_metrics = print_overall_summary(results, pck_thresholds=pck_thresholds)
    elif len(results) == 1 and not single_mode:
        pooled_metrics = results[0]["metrics"]

    # Export summary CSV if specified
    if args.summary_csv:
        summary_path = args.summary_csv.expanduser().resolve()
        export_summary_csv(results, pooled_metrics, summary_path, pck_thresholds=pck_thresholds)
    elif not single_mode:
        # Save a default summary next to the input directory or in PREDICTION_ROOT/Test
        default_summary = PREDICTION_ROOT / "Test" / "accuracy_summary.csv"
        export_summary_csv(results, pooled_metrics, default_summary, pck_thresholds=pck_thresholds)


if __name__ == "__main__":
    main()
