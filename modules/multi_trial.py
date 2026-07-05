"""
VMotionLab - Multi-Trial Analysis
=================================

This module combines normalized curves from multiple VMotionLab projects.

Responsibilities:
- Find projects that contain normalized_curves.csv
- Combine normalized curves from selected trials
- Calculate mean and standard deviation across trials
- Generate multi-trial statistics
- Save mean ± SD figures
- Save multi-trial analysis outputs

This module does not contain Streamlit UI code.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.project_manager import list_recent_projects, load_metadata


NORMALIZED_CURVES_FILENAME = "normalized_curves.csv"

MULTI_TRIAL_COMBINED_FILENAME = "multi_trial_combined_curves.csv"
MULTI_TRIAL_MEAN_SD_FILENAME = "multi_trial_mean_sd.csv"
MULTI_TRIAL_STATISTICS_FILENAME = "multi_trial_statistics.csv"
MULTI_TRIAL_INFO_FILENAME = "multi_trial_info.json"

FIGURES_DIRNAME = "figures"


EXCLUDED_COLUMNS = {
    "percent_movement",
    "time",
    "frame",
    "pose_detected",
}


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


def get_display_label(variable: str) -> str:
    """Return clinician-friendly variable label."""
    return ANGLE_LABELS.get(variable, variable.replace("_", " ").title())


def load_normalized_curves(project_path: Path | str) -> pd.DataFrame:
    """Load normalized_curves.csv from a project folder."""
    project_path = Path(project_path)
    path = project_path / NORMALIZED_CURVES_FILENAME

    if not path.exists():
        raise FileNotFoundError(f"normalized_curves.csv not found: {path}")

    df = pd.read_csv(path)

    if "percent_movement" not in df.columns:
        raise ValueError("normalized_curves.csv must contain `percent_movement`.")

    df["percent_movement"] = pd.to_numeric(df["percent_movement"], errors="coerce")
    df = df.dropna(subset=["percent_movement"]).copy()
    df = df.sort_values("percent_movement").reset_index(drop=True)

    return df


def get_angle_columns_from_normalized(df: pd.DataFrame) -> List[str]:
    """
    Return available angle columns from normalized_curves.csv.

    This version is more robust because some CSV columns may be loaded as
    object/text even though their values are numeric.
    """
    columns = []

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        numeric_values = pd.to_numeric(df[column], errors="coerce")

        if numeric_values.notna().sum() > 0:
            columns.append(column)

    return columns


def find_projects_with_normalized_curves(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Find recent projects that contain normalized_curves.csv.
    """
    recent = list_recent_projects(limit=limit)
    eligible = []

    for metadata in recent:
        project_path = Path(metadata.get("project_path", ""))

        if not project_path.exists():
            continue

        if (project_path / NORMALIZED_CURVES_FILENAME).exists():
            eligible.append(metadata)

    return eligible


def make_project_label(metadata: Dict[str, Any]) -> str:
    """Create readable label for project selection."""
    subject_id = metadata.get("subject_id", "-")
    task = metadata.get("task", "-")
    trial_name = metadata.get("trial_name", "-")
    created_at = metadata.get("created_at", "-")
    project_name = metadata.get("project_name", "-")

    return f"{subject_id} | {task} | {trial_name} | {created_at} | {project_name}"


def get_common_angle_columns(project_paths: List[Path | str]) -> List[str]:
    """
    Return angle columns shared by all selected projects.
    """
    common_columns: Optional[set[str]] = None

    for project_path in project_paths:
        df = load_normalized_curves(project_path)
        columns = set(get_angle_columns_from_normalized(df))

        if common_columns is None:
            common_columns = columns
        else:
            common_columns = common_columns.intersection(columns)

    if common_columns is None:
        return []

    preferred_order = [
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

    ordered = [column for column in preferred_order if column in common_columns]
    remaining = sorted([column for column in common_columns if column not in ordered])

    return ordered + remaining


def combine_normalized_curves(
    project_paths: List[Path | str],
    selected_variables: List[str],
) -> pd.DataFrame:
    """
    Combine normalized curves from multiple projects into long format.

    Output columns:
    - project_name
    - subject_id
    - task
    - trial_name
    - variable
    - joint
    - percent_movement
    - angle_deg
    """
    rows = []

    for project_path in project_paths:
        project_path = Path(project_path)

        metadata = load_metadata(project_path) or {}
        normalized_df = load_normalized_curves(project_path)

        for variable in selected_variables:
            if variable not in normalized_df.columns:
                continue

            temp = pd.DataFrame(
                {
                    "project_name": project_path.name,
                    "subject_id": metadata.get("subject_id", ""),
                    "task": metadata.get("task", ""),
                    "trial_name": metadata.get("trial_name", ""),
                    "variable": variable,
                    "joint": get_display_label(variable),
                    "percent_movement": normalized_df["percent_movement"],
                    "angle_deg": pd.to_numeric(normalized_df[variable], errors="coerce"),
                }
            )

            rows.append(temp)

    if not rows:
        raise ValueError("No normalized curves could be combined.")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(
        ["variable", "percent_movement", "project_name"]
    ).reset_index(drop=True)

    return combined


def calculate_mean_sd(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean and SD across trials at each normalized percent point.
    """
    grouped = (
        combined_df
        .groupby(["variable", "joint", "percent_movement"], as_index=False)
        .agg(
            mean_deg=("angle_deg", "mean"),
            sd_deg=("angle_deg", "std"),
            n_trials=("angle_deg", "count"),
        )
    )

    grouped["mean_deg"] = grouped["mean_deg"].round(3)
    grouped["sd_deg"] = grouped["sd_deg"].fillna(0).round(3)

    return grouped


def calculate_multi_trial_statistics(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trial-level statistics and summarize across trials.

    Output summarizes:
    - mean of trial means
    - SD of trial means
    - mean ROM
    - SD ROM
    - mean maximum angle
    - mean minimum angle
    """
    trial_stats = (
        combined_df
        .groupby(["project_name", "subject_id", "task", "trial_name", "variable", "joint"])
        .agg(
            trial_mean_deg=("angle_deg", "mean"),
            trial_min_deg=("angle_deg", "min"),
            trial_max_deg=("angle_deg", "max"),
        )
        .reset_index()
    )

    trial_stats["trial_rom_deg"] = trial_stats["trial_max_deg"] - trial_stats["trial_min_deg"]

    summary = (
        trial_stats
        .groupby(["variable", "joint"], as_index=False)
        .agg(
            n_trials=("project_name", "count"),
            mean_of_trial_means_deg=("trial_mean_deg", "mean"),
            sd_of_trial_means_deg=("trial_mean_deg", "std"),
            mean_min_deg=("trial_min_deg", "mean"),
            mean_max_deg=("trial_max_deg", "mean"),
            mean_rom_deg=("trial_rom_deg", "mean"),
            sd_rom_deg=("trial_rom_deg", "std"),
        )
    )

    numeric_columns = [
        "mean_of_trial_means_deg",
        "sd_of_trial_means_deg",
        "mean_min_deg",
        "mean_max_deg",
        "mean_rom_deg",
        "sd_rom_deg",
    ]

    for column in numeric_columns:
        summary[column] = summary[column].fillna(0).round(2)

    return summary


def save_mean_sd_figure(
    mean_sd_df: pd.DataFrame,
    variable: str,
    output_path: Path | str,
    show_individual_trials: bool = False,
    combined_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Save mean ± SD figure for one variable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variable_df = mean_sd_df[mean_sd_df["variable"] == variable].copy()

    if variable_df.empty:
        raise ValueError(f"No mean ± SD data found for variable: {variable}")

    x = variable_df["percent_movement"].to_numpy(dtype=float)
    mean = variable_df["mean_deg"].to_numpy(dtype=float)
    sd = variable_df["sd_deg"].to_numpy(dtype=float)

    lower = mean - sd
    upper = mean + sd

    plt.figure(figsize=(10, 6))

    if show_individual_trials and combined_df is not None:
        individual_df = combined_df[combined_df["variable"] == variable].copy()

        for project_name, trial_df in individual_df.groupby("project_name"):
            plt.plot(
                trial_df["percent_movement"],
                trial_df["angle_deg"],
                linewidth=0.8,
                alpha=0.35,
            )

    plt.plot(x, mean, linewidth=2.5, label="Mean")
    plt.fill_between(x, lower, upper, alpha=0.25, label="±1 SD")

    plt.title(f"{get_display_label(variable)} - Mean ± SD")
    plt.xlabel("Movement cycle (%)")
    plt.ylabel("Angle (degrees)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def run_multi_trial_analysis(
    output_project_path: Path | str,
    selected_project_paths: List[Path | str],
    selected_variables: List[str],
    primary_variable: str,
    show_individual_trials: bool = True,
) -> Dict[str, Any]:
    """
    Run multi-trial analysis and save outputs to the active project folder.
    """
    output_project_path = Path(output_project_path)

    if len(selected_project_paths) < 2:
        raise ValueError("Select at least two trials for multi-trial analysis.")

    if not selected_variables:
        raise ValueError("Select at least one angle variable.")

    if primary_variable not in selected_variables:
        raise ValueError("Primary plot variable must be one of the selected variables.")

    figures_dir = output_project_path / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined_df = combine_normalized_curves(
        project_paths=selected_project_paths,
        selected_variables=selected_variables,
    )

    mean_sd_df = calculate_mean_sd(combined_df)
    statistics_df = calculate_multi_trial_statistics(combined_df)

    combined_path = output_project_path / MULTI_TRIAL_COMBINED_FILENAME
    mean_sd_path = output_project_path / MULTI_TRIAL_MEAN_SD_FILENAME
    statistics_path = output_project_path / MULTI_TRIAL_STATISTICS_FILENAME
    info_path = output_project_path / MULTI_TRIAL_INFO_FILENAME

    combined_df.to_csv(combined_path, index=False)
    mean_sd_df.to_csv(mean_sd_path, index=False)
    statistics_df.to_csv(statistics_path, index=False)

    safe_variable_name = primary_variable.lower().replace(" ", "_")
    figure_path = figures_dir / f"multi_trial_mean_sd_{safe_variable_name}.png"

    save_mean_sd_figure(
        mean_sd_df=mean_sd_df,
        variable=primary_variable,
        output_path=figure_path,
        show_individual_trials=show_individual_trials,
        combined_df=combined_df,
    )

    selected_projects_info = []

    for project_path in selected_project_paths:
        project_path = Path(project_path)
        metadata = load_metadata(project_path) or {}

        selected_projects_info.append(
            {
                "project_name": project_path.name,
                "project_path": str(project_path),
                "subject_id": metadata.get("subject_id", ""),
                "task": metadata.get("task", ""),
                "trial_name": metadata.get("trial_name", ""),
                "created_at": metadata.get("created_at", ""),
            }
        )

    info = {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_type": "multi_trial_mean_sd",
        "output_project_path": str(output_project_path),
        "selected_projects": selected_projects_info,
        "n_selected_projects": int(len(selected_project_paths)),
        "selected_variables": selected_variables,
        "primary_variable": primary_variable,
        "show_individual_trials": bool(show_individual_trials),
        "outputs": {
            "combined_curves": MULTI_TRIAL_COMBINED_FILENAME,
            "mean_sd": MULTI_TRIAL_MEAN_SD_FILENAME,
            "statistics": MULTI_TRIAL_STATISTICS_FILENAME,
            "info": MULTI_TRIAL_INFO_FILENAME,
            "figure": str(Path(FIGURES_DIRNAME) / figure_path.name),
        },
        "clinical_note": (
            "Multi-trial mean and standard deviation are calculated from normalized "
            "0-100% movement curves. Values remain sagittal-plane clinical approximations."
        ),
    }

    save_json(info_path, info)

    return info