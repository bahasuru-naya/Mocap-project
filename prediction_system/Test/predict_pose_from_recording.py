"""Generate pose predictions from a recorded data-collection CSV."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

PREDICTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREDICTION_ROOT / "src"))

from system.prediction_engine import (
    IMU_COLS,
    POSE_COLS,
    PoseEnsemble,
)


SAMPLE_PERIOD_SECONDS = 0.05
SAMPLE_PERIOD_TOLERANCE_SECONDS = 0.002


def _read_dataset(path):
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

        rows = []
        timestamps = []
        feature_rows = []
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

    return columns, rows, np.asarray(feature_rows, dtype=np.float32), missing_values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict 17 3-D pose landmarks from a synchronized 20 Hz "
            "data-collection CSV."
        )
    )
    parser.add_argument("dataset", type=Path, help="CSV written by data_collection")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV (default: <dataset>_predictions.csv)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else dataset_path.with_name(f"{dataset_path.stem}_predictions.csv")
    )
    if dataset_path == output_path:
        parser.error("The output path must not overwrite the input dataset.")

    try:
        columns, rows, features, missing_values = _read_dataset(dataset_path)

        base_dir = str(PREDICTION_ROOT)
        model_paths = [
            os.path.join(base_dir, "model", f"best_model_overall_member{i}.onnx")
            for i in range(3)
        ]
        missing_models = [path for path in model_paths if not os.path.isfile(path)]
        if missing_models:
            raise FileNotFoundError(
                "Missing ONNX model file(s): " + ", ".join(missing_models)
            )

        predictor = PoseEnsemble(
            model_paths,
            os.path.join(base_dir, "scalers", "imu_scaler.joblib"),
            os.path.join(base_dir, "scalers", "pose_scaler.joblib"),
            prefer_gpu=False,
        )
        window_size = predictor.window_size
        if len(rows) < window_size:
            raise ValueError(
                f"Dataset has {len(rows)} rows; the loaded model requires "
                f"{window_size} rows per window."
            )

        prediction_columns = [f"pred_{column}" for column in POSE_COLS]
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=[*columns, *prediction_columns],
                extrasaction="ignore",
            )
            writer.writeheader()
            for end_index in range(window_size - 1, len(rows)):
                pose = predictor.predict_window(
                    features[end_index - window_size + 1:end_index + 1]
                )
                output_row = dict(rows[end_index])
                output_row.update({
                    f"pred_cv_{point_index}_{axis}": repr(float(pose[point_index, axis_index]))
                    for point_index in range(17)
                    for axis_index, axis in enumerate(("x", "y", "z"))
                })
                writer.writerow(output_row)

    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Predictions saved to: {output_path}")
    print(f"Input rows: {len(rows)}; predictions: {len(rows) - window_size + 1}")
    print(f"IMU values missing in input: {missing_values}")


if __name__ == "__main__":
    main()
