"""
VMotionLab - Trial Trimmer
==========================

This module trims raw motion and landmark data based on a selected time window.

Responsibilities:
- Load motion_raw.csv
- Select a useful movement window
- Save motion_trimmed.csv
- Optionally save landmarks_trimmed.csv
- Preserve original time/frame information for traceability
- Reset trimmed time so analysis starts at 0 seconds

This module does not contain Streamlit UI code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MOTION_RAW_FILENAME = "motion_raw.csv"
MOTION_TRIMMED_FILENAME = "motion_trimmed.csv"
LANDMARKS_FILENAME = "landmarks.csv"
LANDMARKS_TRIMMED_FILENAME = "landmarks_trimmed.csv"


PREFERRED_PREVIEW_COLUMNS = [
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


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a CSV file and return a pandas DataFrame."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def load_motion_raw(project_path: Path | str) -> pd.DataFrame:
    """Load motion_raw.csv from a project folder."""
    project_path = Path(project_path)
    return load_csv(project_path / MOTION_RAW_FILENAME)


def load_landmarks(project_path: Path | str) -> Optional[pd.DataFrame]:
    """Load landmarks.csv if available."""
    project_path = Path(project_path)
    path = project_path / LANDMARKS_FILENAME

    if not path.exists():
        return None

    return load_csv(path)


def validate_time_column(df: pd.DataFrame, time_column: str = "time") -> None:
    """Validate that a DataFrame contains a numeric time column."""
    if time_column not in df.columns:
        raise ValueError(f"Required column not found: {time_column}")

    if not pd.api.types.is_numeric_dtype(df[time_column]):
        raise ValueError(f"Column must be numeric: {time_column}")


def get_time_bounds(
    df: pd.DataFrame,
    time_column: str = "time",
) -> Tuple[float, float]:
    """Return minimum and maximum time values."""
    validate_time_column(df, time_column=time_column)

    time_values = df[time_column].dropna()

    if time_values.empty:
        raise ValueError("Time column contains no valid values.")

    return float(time_values.min()), float(time_values.max())


def get_preview_columns(df: pd.DataFrame) -> List[str]:
    """
    Return numeric angle columns suitable for preview plotting.

    Preference is given to VMotionLab clinical angle columns.
    """
    columns = [
        column
        for column in PREFERRED_PREVIEW_COLUMNS
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    if columns:
        return columns

    excluded = {
        "frame",
        "frame_original",
        "time",
        "time_original",
        "pose_detected",
    }

    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def _reset_frame_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve original frame as frame_original and reset frame to start at 0.

    This function supports both one-row-per-frame motion data and long-format
    landmark data with multiple rows per frame.
    """
    if "frame" not in df.columns:
        return df

    df = df.copy()

    if "frame_original" not in df.columns:
        df["frame_original"] = df["frame"]

    unique_frames = sorted(df["frame_original"].dropna().unique())
    frame_map = {old_frame: new_index for new_index, old_frame in enumerate(unique_frames)}

    df["frame"] = df["frame_original"].map(frame_map)

    return df


def trim_dataframe_by_time(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
    time_column: str = "time",
    rebase_time_to_zero: bool = True,
    reset_frame: bool = True,
) -> pd.DataFrame:
    """
    Trim a DataFrame by time range.

    Args:
        df:
            Input DataFrame.
        start_time:
            Start time in seconds from the original recording.
        end_time:
            End time in seconds from the original recording.
        time_column:
            Name of the time column.
        rebase_time_to_zero:
            If True, keep original time as time_original and reset time so the
            trimmed file starts at 0 seconds.
        reset_frame:
            If True, keep original frame as frame_original and reset frame so the
            trimmed file starts at 0.

    Returns:
        Trimmed DataFrame.
    """
    validate_time_column(df, time_column=time_column)

    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")

    mask = (df[time_column] >= start_time) & (df[time_column] <= end_time)
    trimmed = df.loc[mask].copy()

    if trimmed.empty:
        raise ValueError("Selected time window contains no data.")

    if rebase_time_to_zero:
        if "time_original" not in trimmed.columns:
            trimmed["time_original"] = trimmed[time_column]

        trimmed[time_column] = trimmed[time_column] - float(start_time)

    if reset_frame:
        trimmed = _reset_frame_column(trimmed)

    return trimmed.reset_index(drop=True)


def summarize_trim(
    raw_motion: pd.DataFrame,
    trimmed_motion: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """Create a dictionary summary for metadata.json."""
    raw_min, raw_max = get_time_bounds(raw_motion)
    trimmed_min, trimmed_max = get_time_bounds(trimmed_motion)

    pose_detection_rate = None

    if "pose_detected" in trimmed_motion.columns and len(trimmed_motion) > 0:
        pose_detection_rate = float(trimmed_motion["pose_detected"].mean() * 100.0)

    return {
        "input_file": MOTION_RAW_FILENAME,
        "output_file": MOTION_TRIMMED_FILENAME,
        "start_time_original_seconds": float(start_time),
        "end_time_original_seconds": float(end_time),
        "selected_window_seconds": float(end_time - start_time),
        "raw_duration_seconds": float(raw_max - raw_min),
        "trimmed_duration_seconds": float(trimmed_max - trimmed_min),
        "raw_rows": int(len(raw_motion)),
        "trimmed_rows": int(len(trimmed_motion)),
        "time_rebased_to_zero": True,
        "frame_reset_to_zero": True,
        "pose_detection_rate_percent": pose_detection_rate,
    }


def trim_trial_files(
    project_path: Path | str,
    start_time: float,
    end_time: float,
    trim_landmarks: bool = True,
) -> Dict[str, Any]:
    """
    Trim VMotionLab project files.

    Required input:
        motion_raw.csv

    Outputs:
        motion_trimmed.csv

    Optional output:
        landmarks_trimmed.csv
    """
    project_path = Path(project_path)

    raw_motion_path = project_path / MOTION_RAW_FILENAME
    trimmed_motion_path = project_path / MOTION_TRIMMED_FILENAME

    raw_motion = load_csv(raw_motion_path)

    trimmed_motion = trim_dataframe_by_time(
        raw_motion,
        start_time=start_time,
        end_time=end_time,
        rebase_time_to_zero=True,
        reset_frame=True,
    )

    trimmed_motion.to_csv(trimmed_motion_path, index=False)

    landmarks_trimmed_saved = False
    landmarks_trimmed_rows = 0

    if trim_landmarks:
        landmarks_path = project_path / LANDMARKS_FILENAME

        if landmarks_path.exists():
            landmarks = load_csv(landmarks_path)

            trimmed_landmarks = trim_dataframe_by_time(
                landmarks,
                start_time=start_time,
                end_time=end_time,
                rebase_time_to_zero=True,
                reset_frame=True,
            )

            trimmed_landmarks_path = project_path / LANDMARKS_TRIMMED_FILENAME
            trimmed_landmarks.to_csv(trimmed_landmarks_path, index=False)

            landmarks_trimmed_saved = True
            landmarks_trimmed_rows = int(len(trimmed_landmarks))

    summary = summarize_trim(
        raw_motion=raw_motion,
        trimmed_motion=trimmed_motion,
        start_time=start_time,
        end_time=end_time,
    )

    summary["landmarks_trimmed_saved"] = bool(landmarks_trimmed_saved)
    summary["landmarks_trimmed_file"] = (
        LANDMARKS_TRIMMED_FILENAME if landmarks_trimmed_saved else None
    )
    summary["landmarks_trimmed_rows"] = int(landmarks_trimmed_rows)

    return summary