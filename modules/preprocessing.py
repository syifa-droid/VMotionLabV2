"""
VMotionLabV2 - Preprocessing
============================

Preprocessing utilities for VMotionLabV2.

Main functions:
- Load motion_raw.csv or motion_trimmed.csv
- Detect available clinical angle columns
- Interpolate missing values
- Apply Butterworth low-pass filtering
- Save motion_filtered.csv
- Save processing_info.json

This module supports filtering multiple joints/angles at the same time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


MOTION_RAW_FILENAME = "motion_raw.csv"
MOTION_TRIMMED_FILENAME = "motion_trimmed.csv"
MOTION_FILTERED_FILENAME = "motion_filtered.csv"
PROCESSING_INFO_FILENAME = "processing_info.json"


EXCLUDED_COLUMNS = {
    "frame",
    "frame_original",
    "time",
    "time_original",
    "timestamp",
    "timestamp_ms",
    "pose_detected",
    "visibility",
    "presence",
}


PREFERRED_CLINICAL_COLUMNS = [
    "trunk_flexion",
    "hip_flexion_r",
    "knee_flexion_r",
    "ankle_angle_r",
    "shoulder_flexion_r",
    "elbow_flexion_r",
    "wrist_flexion_r",
    "hip_flexion_l",
    "knee_flexion_l",
    "ankle_angle_l",
    "shoulder_flexion_l",
    "elbow_flexion_l",
    "wrist_flexion_l",
]


ANGLE_LABELS = {
    "trunk_flexion": "Trunk Flexion",
    "hip_flexion_r": "Right Hip Flexion",
    "knee_flexion_r": "Right Knee Flexion",
    "ankle_angle_r": "Right Ankle Angle",
    "shoulder_flexion_r": "Right Shoulder Flexion",
    "elbow_flexion_r": "Right Elbow Flexion",
    "wrist_flexion_r": "Right Wrist Flexion",
    "hip_flexion_l": "Left Hip Flexion",
    "knee_flexion_l": "Left Knee Flexion",
    "ankle_angle_l": "Left Ankle Angle",
    "shoulder_flexion_l": "Left Shoulder Flexion",
    "elbow_flexion_l": "Left Elbow Flexion",
    "wrist_flexion_l": "Left Wrist Flexion",
}


def save_json(path: Path | str, data: Dict[str, Any]) -> None:
    """Save dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def get_display_label(column: str) -> str:
    """Return clinician-friendly angle label."""
    return ANGLE_LABELS.get(column, column.replace("_", " ").title())


def load_motion_file(path: Path | str) -> pd.DataFrame:
    """Load motion CSV and clean column names."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise RuntimeError(f"{path.name} is empty.")

    if "time" not in df.columns:
        raise ValueError("Motion file must contain a `time` column.")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    if df["time"].isna().all():
        raise ValueError("Column must be numeric: time")

    df["time"] = df["time"].interpolate().bfill().ffill()
    df["time"] = df["time"] - df["time"].iloc[0]

    return df


def estimate_sampling_frequency(df: pd.DataFrame) -> float:
    """Estimate sampling frequency from time column."""
    if "time" not in df.columns:
        return 0.0

    time_values = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy()

    if len(time_values) < 2:
        return 0.0

    diffs = np.diff(time_values)
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 0.0

    median_dt = float(np.median(diffs))

    if median_dt <= 0:
        return 0.0

    return float(1.0 / median_dt)


def get_available_angle_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect available clinical angle columns.

    This function accepts columns that can be converted to numeric values.
    It prioritizes known VMotionLab clinical angle names.
    """
    available = []

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        numeric_values = pd.to_numeric(df[column], errors="coerce")

        if numeric_values.notna().sum() > 0:
            available.append(column)

    ordered = [col for col in PREFERRED_CLINICAL_COLUMNS if col in available]
    remaining = sorted([col for col in available if col not in ordered])

    return ordered + remaining


def interpolate_columns(
    df: pd.DataFrame,
    selected_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Interpolate selected columns only.
    """
    output_df = df.copy()

    missing_before = {}
    missing_after = {}

    for column in selected_columns:
        if column not in output_df.columns:
            continue

        series = pd.to_numeric(output_df[column], errors="coerce")
        missing_before[column] = int(series.isna().sum())

        series = series.interpolate(method="linear").bfill().ffill()

        output_df[column] = series
        missing_after[column] = int(output_df[column].isna().sum())

    info = {
        "selected_columns": selected_columns,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }

    return output_df, info


def butterworth_lowpass_filter(
    values: np.ndarray,
    sampling_frequency: float,
    cutoff_frequency: float,
    filter_order: int,
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to one numeric array.
    """
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")

    nyquist = sampling_frequency / 2.0

    if cutoff_frequency >= nyquist:
        raise ValueError(
            f"Cutoff frequency must be lower than Nyquist frequency. "
            f"Cutoff={cutoff_frequency:.2f} Hz, Nyquist={nyquist:.2f} Hz."
        )

    normal_cutoff = cutoff_frequency / nyquist

    b, a = butter(
        N=filter_order,
        Wn=normal_cutoff,
        btype="low",
        analog=False,
    )

    min_length = max(len(a), len(b)) * 3

    if len(values) <= min_length:
        return values

    return filtfilt(b, a, values)


def filter_columns(
    df: pd.DataFrame,
    selected_columns: List[str],
    sampling_frequency: float,
    cutoff_frequency: float,
    filter_order: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply low-pass filter to selected columns only.
    """
    output_df = df.copy()

    filtered_columns = []
    skipped_columns = {}

    for column in selected_columns:
        if column not in output_df.columns:
            skipped_columns[column] = "Column not found."
            continue

        values = pd.to_numeric(output_df[column], errors="coerce")
        values = values.interpolate().bfill().ffill().to_numpy(dtype=float)

        try:
            filtered_values = butterworth_lowpass_filter(
                values=values,
                sampling_frequency=sampling_frequency,
                cutoff_frequency=cutoff_frequency,
                filter_order=filter_order,
            )

            output_df[column] = filtered_values
            filtered_columns.append(column)

        except Exception as exc:
            skipped_columns[column] = str(exc)
            output_df[column] = values

    info = {
        "filtered_columns": filtered_columns,
        "skipped_columns": skipped_columns,
    }

    return output_df, info


def preprocess_motion_dataframe(
    df: pd.DataFrame,
    selected_columns: List[str],
    apply_interpolation: bool = True,
    apply_filter: bool = True,
    cutoff_frequency: float = 6.0,
    filter_order: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess selected clinical angle columns.

    Only selected columns are interpolated/filtered.
    Other columns are preserved.
    """
    if not selected_columns:
        raise ValueError("Select at least one joint/angle column to preprocess.")

    output_df = df.copy()

    sampling_frequency = estimate_sampling_frequency(output_df)

    if sampling_frequency <= 0:
        raise ValueError("Could not estimate sampling frequency from time column.")

    nyquist = sampling_frequency / 2.0

    if apply_filter and cutoff_frequency >= nyquist:
        raise ValueError(
            f"Cutoff frequency must be lower than Nyquist frequency. "
            f"Current sampling frequency is {sampling_frequency:.2f} Hz, "
            f"Nyquist is {nyquist:.2f} Hz."
        )

    interpolation_info = {}
    filtering_info = {}

    if apply_interpolation:
        output_df, interpolation_info = interpolate_columns(
            df=output_df,
            selected_columns=selected_columns,
        )

    if apply_filter:
        output_df, filtering_info = filter_columns(
            df=output_df,
            selected_columns=selected_columns,
            sampling_frequency=sampling_frequency,
            cutoff_frequency=cutoff_frequency,
            filter_order=filter_order,
        )

    info = {
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "sampling_frequency_hz": round(float(sampling_frequency), 3),
        "nyquist_frequency_hz": round(float(nyquist), 3),
        "selected_columns": selected_columns,
        "selected_labels": [get_display_label(col) for col in selected_columns],
        "apply_interpolation": bool(apply_interpolation),
        "apply_filter": bool(apply_filter),
        "cutoff_frequency_hz": float(cutoff_frequency),
        "filter_order": int(filter_order),
        "interpolation": interpolation_info,
        "filtering": filtering_info,
        "n_rows": int(len(output_df)),
        "duration_seconds": float(output_df["time"].max() - output_df["time"].min()),
        "note": (
            "Preprocessing was applied only to selected clinical angle columns. "
            "Unselected columns were preserved but not filtered."
        ),
    }

    return output_df, info


def preprocess_motion_file(
    input_path: Path | str,
    output_path: Path | str,
    info_path: Path | str,
    selected_columns: Optional[List[str]] = None,
    apply_interpolation: bool = True,
    apply_filter: bool = True,
    cutoff_frequency: float = 6.0,
    filter_order: int = 4,
) -> Dict[str, Any]:
    """
    Preprocess motion CSV file and save output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    info_path = Path(info_path)

    df = load_motion_file(input_path)

    available_columns = get_available_angle_columns(df)

    if selected_columns is None:
        selected_columns = available_columns

    selected_columns = [
        column for column in selected_columns
        if column in available_columns
    ]

    if not selected_columns:
        raise ValueError("No valid selected angle columns found.")

    filtered_df, info = preprocess_motion_dataframe(
        df=df,
        selected_columns=selected_columns,
        apply_interpolation=apply_interpolation,
        apply_filter=apply_filter,
        cutoff_frequency=cutoff_frequency,
        filter_order=filter_order,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(output_path, index=False)

    info["input_file"] = str(input_path)
    info["output_file"] = str(output_path)
    info["info_file"] = str(info_path)

    save_json(info_path, info)

    return info