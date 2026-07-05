"""
VMotionLab - Kinematics
=======================

This module handles clinical kinematics analysis.

Responsibilities:
- Load raw, trimmed, or filtered motion data
- Calculate clinician-friendly statistics
- Normalize curves to 0-100% movement cycle
- Save statistics.csv
- Save normalized_curves.csv
- Save kinematic figures into figures/
- Return analysis summary for metadata.json

This module does not contain Streamlit UI code.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOTION_RAW_FILENAME = "motion_raw.csv"
MOTION_TRIMMED_FILENAME = "motion_trimmed.csv"
MOTION_FILTERED_FILENAME = "motion_filtered.csv"

STATISTICS_FILENAME = "statistics.csv"
NORMALIZED_CURVES_FILENAME = "normalized_curves.csv"
KINEMATICS_INFO_FILENAME = "kinematics_info.json"

FIGURES_DIRNAME = "figures"


EXCLUDED_NUMERIC_COLUMNS = {
    "frame",
    "frame_original",
    "time",
    "time_original",
    "pose_detected",
}


PREFERRED_ANGLE_COLUMNS = [
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
    """Save dictionary as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def load_motion_file(path: Path | str) -> pd.DataFrame:
    """Load a VMotionLab motion CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("Motion file must contain a `time` column.")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        raise ValueError("Motion file contains no valid time values.")

    return df


def get_available_kinematics_inputs(project_path: Path | str) -> Dict[str, Path]:
    """
    Return available kinematics input files.

    Recommended priority:
        Filtered motion
        Trimmed motion
        Raw motion
    """
    project_path = Path(project_path)

    available: Dict[str, Path] = {}

    filtered_path = project_path / MOTION_FILTERED_FILENAME
    trimmed_path = project_path / MOTION_TRIMMED_FILENAME
    raw_path = project_path / MOTION_RAW_FILENAME

    if filtered_path.exists():
        available["Filtered motion"] = filtered_path

    if trimmed_path.exists():
        available["Trimmed motion"] = trimmed_path

    if raw_path.exists():
        available["Raw motion"] = raw_path

    return available


def choose_default_input_label(available_inputs: Dict[str, Path]) -> Optional[str]:
    """Return the recommended default input label."""
    if "Filtered motion" in available_inputs:
        return "Filtered motion"

    if "Trimmed motion" in available_inputs:
        return "Trimmed motion"

    if "Raw motion" in available_inputs:
        return "Raw motion"

    return None


def get_angle_columns(df: pd.DataFrame) -> List[str]:
    """
    Return clinical angle columns available for analysis.
    """
    preferred = [
        column
        for column in PREFERRED_ANGLE_COLUMNS
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    if preferred:
        return preferred

    fallback = []

    for column in df.columns:
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            fallback.append(column)

    return fallback


def get_display_label(column: str) -> str:
    """Return a clinician-friendly label for an angle column."""
    return ANGLE_LABELS.get(column, column.replace("_", " ").title())


def calculate_statistics(
    df: pd.DataFrame,
    angle_columns: List[str],
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for selected angle columns.

    Output columns:
    - joint
    - variable
    - valid_samples
    - missing_samples
    - minimum_deg
    - maximum_deg
    - mean_deg
    - sd_deg
    - median_deg
    - rom_deg
    - initial_deg
    - final_deg
    """
    rows: List[Dict[str, Any]] = []

    for column in angle_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.dropna()

        if valid.empty:
            rows.append(
                {
                    "joint": get_display_label(column),
                    "variable": column,
                    "valid_samples": 0,
                    "missing_samples": int(values.isna().sum()),
                    "minimum_deg": np.nan,
                    "maximum_deg": np.nan,
                    "mean_deg": np.nan,
                    "sd_deg": np.nan,
                    "median_deg": np.nan,
                    "rom_deg": np.nan,
                    "initial_deg": np.nan,
                    "final_deg": np.nan,
                }
            )
            continue

        minimum = float(valid.min())
        maximum = float(valid.max())

        rows.append(
            {
                "joint": get_display_label(column),
                "variable": column,
                "valid_samples": int(valid.count()),
                "missing_samples": int(values.isna().sum()),
                "minimum_deg": minimum,
                "maximum_deg": maximum,
                "mean_deg": float(valid.mean()),
                "sd_deg": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                "median_deg": float(valid.median()),
                "rom_deg": float(maximum - minimum),
                "initial_deg": float(valid.iloc[0]),
                "final_deg": float(valid.iloc[-1]),
            }
        )

    stats_df = pd.DataFrame(rows)

    numeric_columns = [
        "minimum_deg",
        "maximum_deg",
        "mean_deg",
        "sd_deg",
        "median_deg",
        "rom_deg",
        "initial_deg",
        "final_deg",
    ]

    for column in numeric_columns:
        if column in stats_df.columns:
            stats_df[column] = stats_df[column].round(2)

    return stats_df


def normalize_curve(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    points: int = 101,
) -> np.ndarray:
    """
    Normalize a signal to 0-100% movement cycle.

    Args:
        time_values:
            Original time values.
        signal_values:
            Angle signal values.
        points:
            Number of normalized points. Default 101 gives 0-100%.

    Returns:
        Normalized signal array.
    """
    time_values = np.asarray(time_values, dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)

    valid_mask = np.isfinite(time_values) & np.isfinite(signal_values)

    time_valid = time_values[valid_mask]
    signal_valid = signal_values[valid_mask]

    if len(time_valid) < 2:
        return np.full(points, np.nan)

    unique_time, unique_indices = np.unique(time_valid, return_index=True)
    unique_signal = signal_valid[unique_indices]

    if len(unique_time) < 2:
        return np.full(points, np.nan)

    normalized_time = np.linspace(unique_time.min(), unique_time.max(), points)

    return np.interp(normalized_time, unique_time, unique_signal)


def create_normalized_curves(
    df: pd.DataFrame,
    angle_columns: List[str],
    points: int = 101,
) -> pd.DataFrame:
    """
    Create normalized 0-100% movement curves for selected angle columns.
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a `time` column.")

    normalized_percent = np.linspace(0, 100, points)

    output = pd.DataFrame(
        {
            "percent_movement": normalized_percent,
        }
    )

    time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy()

    for column in angle_columns:
        signal_values = pd.to_numeric(df[column], errors="coerce").to_numpy()
        output[column] = normalize_curve(
            time_values=time_values,
            signal_values=signal_values,
            points=points,
        )

    return output


def save_time_curve_figure(
    df: pd.DataFrame,
    angle_columns: List[str],
    output_path: Path | str,
    title: str = "Clinical Joint Angles Over Time",
) -> Path:
    """Save a time-series angle figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for column in angle_columns:
        plt.plot(df["time"], df[column], label=get_display_label(column))

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_normalized_curve_figure(
    normalized_df: pd.DataFrame,
    angle_columns: List[str],
    output_path: Path | str,
    title: str = "Normalized Clinical Joint Angles",
) -> Path:
    """Save a normalized 0-100% movement curve figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for column in angle_columns:
        plt.plot(
            normalized_df["percent_movement"],
            normalized_df[column],
            label=get_display_label(column),
        )

    plt.title(title)
    plt.xlabel("Movement cycle (%)")
    plt.ylabel("Angle (degrees)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_individual_angle_figures(
    df: pd.DataFrame,
    angle_columns: List[str],
    figures_dir: Path | str,
) -> List[str]:
    """
    Save one time-series figure for each selected angle column.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []

    for column in angle_columns:
        safe_name = column.lower().replace(" ", "_")
        output_path = figures_dir / f"{safe_name}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(df["time"], df[column])
        plt.title(get_display_label(column))
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        saved_files.append(str(output_path.name))

    return saved_files


def run_kinematics_analysis(
    project_path: Path | str,
    input_path: Path | str,
    selected_angle_columns: List[str],
    normalize_points: int = 101,
    save_individual_figures: bool = True,
) -> Dict[str, Any]:
    """
    Run kinematics analysis and save outputs.

    Outputs:
        statistics.csv
        normalized_curves.csv
        kinematics_info.json
        figures/kinematics_time_curves.png
        figures/kinematics_normalized_curves.png
        optional individual figures
    """
    project_path = Path(project_path)
    input_path = Path(input_path)

    df = load_motion_file(input_path)

    if not selected_angle_columns:
        raise ValueError("No angle columns selected.")

    missing_columns = [
        column for column in selected_angle_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Selected columns not found: {missing_columns}")

    figures_dir = project_path / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    statistics_df = calculate_statistics(
        df=df,
        angle_columns=selected_angle_columns,
    )

    normalized_df = create_normalized_curves(
        df=df,
        angle_columns=selected_angle_columns,
        points=int(normalize_points),
    )

    statistics_path = project_path / STATISTICS_FILENAME
    normalized_path = project_path / NORMALIZED_CURVES_FILENAME
    kinematics_info_path = project_path / KINEMATICS_INFO_FILENAME

    statistics_df.to_csv(statistics_path, index=False)
    normalized_df.to_csv(normalized_path, index=False)

    time_figure_path = save_time_curve_figure(
        df=df,
        angle_columns=selected_angle_columns,
        output_path=figures_dir / "kinematics_time_curves.png",
    )

    normalized_figure_path = save_normalized_curve_figure(
        normalized_df=normalized_df,
        angle_columns=selected_angle_columns,
        output_path=figures_dir / "kinematics_normalized_curves.png",
    )

    individual_figures: List[str] = []

    if save_individual_figures:
        individual_figures = save_individual_angle_figures(
            df=df,
            angle_columns=selected_angle_columns,
            figures_dir=figures_dir,
        )

    duration_seconds = float(df["time"].max() - df["time"].min())

    summary: Dict[str, Any] = {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": input_path.name,
        "statistics_file": STATISTICS_FILENAME,
        "normalized_curves_file": NORMALIZED_CURVES_FILENAME,
        "kinematics_info_file": KINEMATICS_INFO_FILENAME,
        "figures": {
            "time_curves": str(Path(FIGURES_DIRNAME) / time_figure_path.name),
            "normalized_curves": str(Path(FIGURES_DIRNAME) / normalized_figure_path.name),
            "individual_figures": [
                str(Path(FIGURES_DIRNAME) / filename)
                for filename in individual_figures
            ],
        },
        "rows": int(len(df)),
        "duration_seconds": duration_seconds,
        "selected_angle_columns": selected_angle_columns,
        "normalize_points": int(normalize_points),
        "clinical_angle_convention": {
            "plane_assumption": "sagittal",
            "angle_type": "clinical sagittal-plane approximation",
            "neutral_reference": {
                "hip": "0 degrees",
                "knee": "0 degrees",
                "shoulder": "0 degrees",
                "elbow": "0 degrees",
                "wrist": "0 degrees",
                "trunk": "0 degrees",
                "ankle": "approximately 90 degrees",
            },
        },
        "limitation_note": (
            "Angles are derived from markerless 2D landmarks and assume sagittal-plane movement. "
            "They should not be interpreted as full three-dimensional joint kinematics."
        ),
    }

    save_json(kinematics_info_path, summary)

    return summary