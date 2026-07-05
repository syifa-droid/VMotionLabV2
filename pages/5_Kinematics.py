from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from modules.app_style import apply_global_style
from modules.project_manager import (
    file_exists,
    find_completed_projects,
    get_current_projects,
    load_metadata,
    save_metadata,
    set_current_projects,
)
from modules.preprocessing import (
    MOTION_FILTERED_FILENAME,
    get_available_angle_columns,
    get_display_label,
)
from modules.state import initialize_state, require_active_projects
from modules.ui_navigation import (
    show_workflow_status,
    workflow_navigation,
)


STATISTICS_FILENAME = "statistics.csv"
NORMALIZED_CURVES_FILENAME = "normalized_curves.csv"
KINEMATICS_INFO_FILENAME = "kinematics_info.json"

MULTI_TRIAL_COMBINED_FILENAME = "multi_trial_combined_curves.csv"
MULTI_TRIAL_MEAN_SD_FILENAME = "multi_trial_mean_sd.csv"
MULTI_TRIAL_STATISTICS_FILENAME = "multi_trial_statistics.csv"
MULTI_TRIAL_INFO_FILENAME = "multi_trial_info.json"


st.set_page_config(
    page_title="Kinematics | VMotionLabV2",
    layout="wide",
)

apply_global_style()
initialize_state()

# -----------------------------------------------------------------------------
# Select trials directly from Kinematics page
# -----------------------------------------------------------------------------

completed_projects = find_completed_projects(
    required_files=["motion_filtered.csv"]
)

if not completed_projects:
    st.warning("No preprocessed trials found. Complete Preprocessing first.")

    if st.button("Go to Preprocessing", type="primary"):
        st.switch_page("pages/4_Preprocessing.py")

    st.stop()


labels = []

for metadata in completed_projects:
    subject_id = metadata.get("subject_id", "Unknown")
    task = metadata.get("task", "Unknown")
    trial_name = metadata.get("trial_name", metadata.get("project_name", "Trial"))
    project_name = metadata.get("project_name", "")

    labels.append(
        f"{subject_id} | {task} | {trial_name} | {project_name}"
    )

label_to_metadata = {
    label: metadata
    for label, metadata in zip(labels, completed_projects)
}

current_project_paths = get_current_projects()
current_project_paths_str = [str(path) for path in current_project_paths]

default_labels = []

for label, metadata in label_to_metadata.items():
    if str(metadata.get("project_path")) in current_project_paths_str:
        default_labels.append(label)

if not default_labels and labels:
    default_labels = [labels[0]]

st.divider()
st.markdown("### Select Trial(s) for Kinematic Analysis")

selected_trial_labels = st.multiselect(
    "Choose one or more preprocessed trials",
    options=labels,
    default=default_labels,
    key="kinematics_trial_selector",
)

if not selected_trial_labels:
    st.warning("Select at least one trial.")
    st.stop()

active_project_paths = [
    Path(label_to_metadata[label]["project_path"])
    for label in selected_trial_labels
]

set_current_projects(active_project_paths)

st.session_state["active_project_path"] = str(active_project_paths[0])
st.session_state["active_project_paths"] = [
    str(path) for path in active_project_paths
]

if len(active_project_paths) == 1:
    st.info("Single-trial kinematics mode.")
else:
    st.success(
        f"Multi-trial analysis mode enabled: {len(active_project_paths)} trials selected."
    )

st.title("Kinematics")
st.caption(
    "Analyze one or more preprocessed trials. Selecting multiple trials enables multi-trial comparison."
)

st.info(
    "Kinematics will use all selected active trials. "
    "If more than one trial is selected, VMotionLabV2 will also create comparison and mean ± SD outputs."
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def save_json(path: Path | str, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def load_filtered_motion(project_path: Path) -> pd.DataFrame:
    path = project_path / MOTION_FILTERED_FILENAME

    if not path.exists():
        raise FileNotFoundError(f"{MOTION_FILTERED_FILENAME} not found in {project_path.name}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    if "time" not in df.columns:
        raise ValueError("motion_filtered.csv must contain a time column.")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["time"] = df["time"].interpolate().bfill().ffill()
    df["time"] = df["time"] - df["time"].iloc[0]

    return df


def calculate_statistics(df: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
    rows = []

    for column in selected_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if values.empty:
            continue

        rows.append(
            {
                "variable": column,
                "label": get_display_label(column),
                "n": int(values.count()),
                "min_deg": float(values.min()),
                "max_deg": float(values.max()),
                "mean_deg": float(values.mean()),
                "sd_deg": float(values.std()),
                "rom_deg": float(values.max() - values.min()),
                "initial_deg": float(values.iloc[0]),
                "final_deg": float(values.iloc[-1]),
            }
        )

    return pd.DataFrame(rows)


def normalize_curves(
    df: pd.DataFrame,
    selected_columns: list[str],
    n_points: int = 101,
) -> pd.DataFrame:
    output = pd.DataFrame()
    output["percent_cycle"] = np.linspace(0, 100, n_points)

    time_values = pd.to_numeric(df["time"], errors="coerce")
    time_values = time_values.interpolate().bfill().ffill().to_numpy(dtype=float)

    if len(time_values) < 2:
        raise ValueError("Not enough time points for normalization.")

    normalized_time = np.linspace(time_values[0], time_values[-1], n_points)

    for column in selected_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.interpolate().bfill().ffill().to_numpy(dtype=float)

        if len(values) != len(time_values):
            continue

        output[column] = np.interp(
            normalized_time,
            time_values,
            values,
        )

    return output

def save_kinematics_figures(
    project_path: Path,
    df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    selected_columns: list[str],
) -> dict:
    """
    Save required kinematics PNG figures for the report page.
    """
    figures_dir = project_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    time_figure_path = figures_dir / "kinematics_time_curves.png"
    normalized_figure_path = figures_dir / "kinematics_normalized_curves.png"

    # Time-domain figure
    plt.figure(figsize=(10, 6))

    for column in selected_columns:
        if column not in df.columns:
            continue

        plt.plot(
            df["time"],
            pd.to_numeric(df[column], errors="coerce"),
            label=get_display_label(column),
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title("Kinematic Time Curves")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(time_figure_path, dpi=300)
    plt.close()

    # Normalized movement-cycle figure
    plt.figure(figsize=(10, 6))

    for column in selected_columns:
        if column not in normalized_df.columns:
            continue

        plt.plot(
            normalized_df["percent_cycle"],
            pd.to_numeric(normalized_df[column], errors="coerce"),
            label=get_display_label(column),
        )

    plt.xlabel("Movement cycle (%)")
    plt.ylabel("Angle (degrees)")
    plt.title("Normalized Kinematic Curves")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(normalized_figure_path, dpi=300)
    plt.close()

    return {
        "time_figure": "figures/kinematics_time_curves.png",
        "normalized_figure": "figures/kinematics_normalized_curves.png",
    }


def make_trial_label(metadata: dict, project_path: Path) -> str:
    subject_id = metadata.get("subject_id", "Unknown")
    task = metadata.get("task", "Unknown")
    trial_name = metadata.get("trial_name", project_path.name)

    return f"{subject_id} | {task} | {trial_name}"


def save_single_trial_outputs(
    project_path: Path,
    metadata: dict,
    df: pd.DataFrame,
    selected_columns: list[str],
    n_points: int,
) -> dict:
    statistics_df = calculate_statistics(df, selected_columns)
    normalized_df = normalize_curves(df, selected_columns, n_points=n_points)

    statistics_path = project_path / STATISTICS_FILENAME
    normalized_path = project_path / NORMALIZED_CURVES_FILENAME
    info_path = project_path / KINEMATICS_INFO_FILENAME

    statistics_df.to_csv(statistics_path, index=False)
    normalized_df.to_csv(normalized_path, index=False)

    figure_files = save_kinematics_figures(
        project_path=project_path,
        df=df,
        normalized_df=normalized_df,
        selected_columns=selected_columns,
    )

    info = {
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "selected_columns": selected_columns,
        "selected_labels": [get_display_label(col) for col in selected_columns],
        "n_points": n_points,
        "input_file": MOTION_FILTERED_FILENAME,
        "statistics_file": STATISTICS_FILENAME,
        "normalized_curves_file": NORMALIZED_CURVES_FILENAME,
        "duration_seconds": float(df["time"].max() - df["time"].min()),
        "n_rows": int(len(df)),
        "figures": figure_files,
    }

    save_json(info_path, info)

    metadata["current_step"] = "kinematics"
    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if "kinematics" not in metadata:
        metadata["kinematics"] = {}

    metadata["kinematics"]["single_trial"] = info

    if "files" not in metadata:
        metadata["files"] = {}

    metadata["files"]["statistics"] = STATISTICS_FILENAME
    metadata["files"]["normalized_curves"] = NORMALIZED_CURVES_FILENAME
    metadata["files"]["kinematics_info"] = KINEMATICS_INFO_FILENAME
    metadata["files"]["kinematics_time_figure"] = figure_files["time_figure"]
    metadata["files"]["kinematics_normalized_figure"] = figure_files["normalized_figure"]

    save_metadata(project_path, metadata)

    return info


def build_combined_normalized_curves(
    loaded_trials: list[dict],
    selected_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for trial in loaded_trials:
        project_path = trial["project_path"]
        metadata = trial["metadata"]
        label = make_trial_label(metadata, project_path)

        normalized_path = project_path / NORMALIZED_CURVES_FILENAME

        if not normalized_path.exists():
            continue

        normalized_df = pd.read_csv(normalized_path)

        for _, row in normalized_df.iterrows():
            for column in selected_columns:
                if column not in normalized_df.columns:
                    continue

                rows.append(
                    {
                        "participant_id": metadata.get("subject_id", "Unknown"),
                        "movement_task": metadata.get("task", "Unknown"),
                        "trial_name": metadata.get("trial_name", project_path.name),
                        "project_name": project_path.name,
                        "trial_label": label,
                        "percent_cycle": float(row["percent_cycle"]),
                        "variable": column,
                        "label": get_display_label(column),
                        "value_deg": float(row[column]),
                    }
                )

    return pd.DataFrame(rows)


def build_mean_sd_curves(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()

    mean_sd_df = (
        combined_df
        .groupby(["percent_cycle", "variable", "label"], as_index=False)
        .agg(
            mean_deg=("value_deg", "mean"),
            sd_deg=("value_deg", "std"),
            n_trials=("value_deg", "count"),
        )
    )

    mean_sd_df["sd_deg"] = mean_sd_df["sd_deg"].fillna(0)

    return mean_sd_df


def build_multi_trial_statistics(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()

    rows = []

    for variable, variable_df in combined_df.groupby("variable"):
        label = get_display_label(variable)

        trial_stats = []

        for trial_label, trial_df in variable_df.groupby("trial_label"):
            values = trial_df["value_deg"].dropna()

            if values.empty:
                continue

            trial_stats.append(
                {
                    "trial_label": trial_label,
                    "min_deg": values.min(),
                    "max_deg": values.max(),
                    "mean_deg": values.mean(),
                    "rom_deg": values.max() - values.min(),
                }
            )

        if not trial_stats:
            continue

        stats_df = pd.DataFrame(trial_stats)

        rows.append(
            {
                "variable": variable,
                "label": label,
                "n_trials": int(len(stats_df)),
                "mean_rom_deg": float(stats_df["rom_deg"].mean()),
                "sd_rom_deg": float(stats_df["rom_deg"].std()),
                "mean_angle_mean_deg": float(stats_df["mean_deg"].mean()),
                "sd_angle_mean_deg": float(stats_df["mean_deg"].std()),
                "min_across_trials_deg": float(stats_df["min_deg"].min()),
                "max_across_trials_deg": float(stats_df["max_deg"].max()),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load selected trials
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Selected Trial(s)")

trial_rows = []
loaded_trials = []

for project_path in active_project_paths:
    project_path = Path(project_path)
    metadata = load_metadata(project_path) or {}

    try:
        df = load_filtered_motion(project_path)
        angle_columns = get_available_angle_columns(df)

        loaded_trials.append(
            {
                "project_path": project_path,
                "metadata": metadata,
                "df": df,
                "angle_columns": angle_columns,
            }
        )

        trial_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
                "motion_filtered.csv": "Found",
                "Rows": len(df),
                "Duration (s)": round(float(df["time"].max() - df["time"].min()), 2),
            }
        )

    except Exception as exc:
        trial_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
                "motion_filtered.csv": f"Error: {exc}",
                "Rows": "",
                "Duration (s)": "",
            }
        )

st.dataframe(
    pd.DataFrame(trial_rows),
    use_container_width=True,
    hide_index=True,
)

if not loaded_trials:
    st.error("No selected trial has a valid motion_filtered.csv file.")
    st.stop()


# -----------------------------------------------------------------------------
# Select common joints / angles
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Select Joint / Clinical Angle")

common_columns = set(loaded_trials[0]["angle_columns"])

for trial in loaded_trials[1:]:
    common_columns = common_columns.intersection(set(trial["angle_columns"]))

preferred_order = loaded_trials[0]["angle_columns"]

common_columns = [
    column for column in preferred_order
    if column in common_columns
]

if not common_columns:
    st.error(
        "No common clinical angle columns were found across selected trials. "
        "Please select trials with the same filtered joint angle columns."
    )
    st.stop()

selected_columns = st.multiselect(
    "Select one or more joints / clinical angles",
    options=common_columns,
    default=common_columns[: min(3, len(common_columns))],
    format_func=get_display_label,
    key="kinematics_selected_columns_multi_trial",
)

if not selected_columns:
    st.warning("Select at least one joint/angle.")
    st.stop()

n_points = st.number_input(
    "Normalize each trial to number of points",
    min_value=51,
    max_value=501,
    value=101,
    step=10,
    key="kinematics_normalization_points",
)

n_points = int(n_points)


# -----------------------------------------------------------------------------
# Preview
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Kinematic Preview")

primary_variable = st.selectbox(
    "Primary joint/angle for preview",
    options=selected_columns,
    format_func=get_display_label,
    key="kinematics_primary_variable",
)

fig = go.Figure()

for trial in loaded_trials:
    project_path = trial["project_path"]
    metadata = trial["metadata"]
    df = trial["df"]

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=pd.to_numeric(df[primary_variable], errors="coerce"),
            mode="lines",
            name=make_trial_label(metadata, project_path),
        )
    )

fig.update_layout(
    title=f"{get_display_label(primary_variable)} - Selected Trial(s)",
    xaxis_title="Time (s)",
    yaxis_title="Angle (degrees)",
    height=550,
    autosize=True,
    font=dict(
        family="Poppins, Arial, Helvetica, sans-serif",
        size=12,
    ),
    margin=dict(l=70, r=30, t=80, b=130),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
    ),
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"responsive": True},
)


# -----------------------------------------------------------------------------
# Run kinematics
# -----------------------------------------------------------------------------

st.divider()

if len(active_project_paths) == 1:
    st.markdown("### Run Single-Trial Kinematics")
else:
    st.markdown("### Run Multi-Trial Kinematics Analysis")
    st.info(
        "VMotionLabV2 will calculate individual trial statistics, normalized curves, "
        "combined curves, mean ± SD curves, and multi-trial summary statistics."
    )

if st.button(
    "Run Kinematics for Selected Trial(s)",
    type="primary",
    use_container_width=True,
):
    progress = st.progress(0.0)
    status_box = st.empty()

    results = []
    failures = []

    total_trials = len(loaded_trials)

    for index, trial in enumerate(loaded_trials):
        project_path = trial["project_path"]
        metadata = trial["metadata"]
        df = trial["df"]

        status_box.info(
            f"Processing kinematics {index + 1}/{total_trials}: `{project_path.name}`"
        )

        try:
            info = save_single_trial_outputs(
                project_path=project_path,
                metadata=metadata,
                df=df,
                selected_columns=selected_columns,
                n_points=n_points,
            )

            results.append(
                {
                    "Project": project_path.name,
                    "Status": "Completed",
                    "Statistics": STATISTICS_FILENAME,
                    "Normalized curves": NORMALIZED_CURVES_FILENAME,
                    "Selected angles": len(selected_columns),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "Project": project_path.name,
                    "Status": "Failed",
                    "Reason": str(exc),
                }
            )

        progress.progress((index + 1) / total_trials)

    primary_project_path = loaded_trials[0]["project_path"]

    if len(loaded_trials) > 1 and not failures:
        try:
            combined_df = build_combined_normalized_curves(
                loaded_trials=loaded_trials,
                selected_columns=selected_columns,
            )

            mean_sd_df = build_mean_sd_curves(combined_df)
            multi_stats_df = build_multi_trial_statistics(combined_df)

            combined_df.to_csv(
                primary_project_path / MULTI_TRIAL_COMBINED_FILENAME,
                index=False,
            )

            mean_sd_df.to_csv(
                primary_project_path / MULTI_TRIAL_MEAN_SD_FILENAME,
                index=False,
            )

            multi_stats_df.to_csv(
                primary_project_path / MULTI_TRIAL_STATISTICS_FILENAME,
                index=False,
            )

            multi_info = {
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "kinematics_multi_trial_comparison",
                "primary_project": primary_project_path.name,
                "n_trials": len(loaded_trials),
                "selected_columns": selected_columns,
                "selected_labels": [get_display_label(col) for col in selected_columns],
                "n_points": n_points,
                "trial_projects": [
                    str(trial["project_path"]) for trial in loaded_trials
                ],
                "combined_curves_file": MULTI_TRIAL_COMBINED_FILENAME,
                "mean_sd_file": MULTI_TRIAL_MEAN_SD_FILENAME,
                "statistics_file": MULTI_TRIAL_STATISTICS_FILENAME,
            }

            save_json(primary_project_path / MULTI_TRIAL_INFO_FILENAME, multi_info)

            primary_metadata = load_metadata(primary_project_path) or {}

            if "files" not in primary_metadata:
                primary_metadata["files"] = {}

            primary_metadata["files"]["multi_trial_combined_curves"] = MULTI_TRIAL_COMBINED_FILENAME
            primary_metadata["files"]["multi_trial_mean_sd"] = MULTI_TRIAL_MEAN_SD_FILENAME
            primary_metadata["files"]["multi_trial_statistics"] = MULTI_TRIAL_STATISTICS_FILENAME
            primary_metadata["files"]["multi_trial_info"] = MULTI_TRIAL_INFO_FILENAME

            primary_metadata["current_step"] = "kinematics"
            primary_metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "kinematics" not in primary_metadata:
                primary_metadata["kinematics"] = {}

            primary_metadata["kinematics"]["multi_trial_comparison"] = multi_info

            save_metadata(primary_project_path, primary_metadata)

        except Exception as exc:
            failures.append(
                {
                    "Project": primary_project_path.name,
                    "Status": "Multi-trial comparison failed",
                    "Reason": str(exc),
                }
            )

    status_box.success("Kinematics finished.")

    if results:
        st.success(f"{len(results)} trial(s) processed successfully.")
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True,
        )

    if failures:
        st.error(f"{len(failures)} issue(s) found.")
        st.dataframe(
            pd.DataFrame(failures),
            use_container_width=True,
            hide_index=True,
        )

    st.rerun()


# -----------------------------------------------------------------------------
# Output status
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Kinematics Output Status")

output_rows = []

for trial in loaded_trials:
    project_path = trial["project_path"]
    metadata = trial["metadata"]

    output_rows.append(
        {
            "Participant": metadata.get("subject_id", "Unknown"),
            "Movement / Task": metadata.get("task", "Unknown"),
            "Trial": metadata.get("trial_name", project_path.name),
            "Project": project_path.name,
            "statistics.csv": "Yes" if file_exists(project_path, STATISTICS_FILENAME) else "No",
            "normalized_curves.csv": "Yes" if file_exists(project_path, NORMALIZED_CURVES_FILENAME) else "No",
        }
    )

st.dataframe(
    pd.DataFrame(output_rows),
    use_container_width=True,
    hide_index=True,
)

can_continue = all(
    file_exists(trial["project_path"], STATISTICS_FILENAME)
    and file_exists(trial["project_path"], NORMALIZED_CURVES_FILENAME)
    for trial in loaded_trials
)

# -----------------------------------------------------------------------------
# Results preview after processing
# -----------------------------------------------------------------------------

primary_project_path = loaded_trials[0]["project_path"]

mean_sd_path = primary_project_path / MULTI_TRIAL_MEAN_SD_FILENAME
combined_path = primary_project_path / MULTI_TRIAL_COMBINED_FILENAME

if len(loaded_trials) > 1 and mean_sd_path.exists():
    st.divider()
    st.markdown("### Multi-Trial Mean ± SD Preview")

    mean_sd_df = pd.read_csv(mean_sd_path)

    if combined_path.exists():
        combined_df = pd.read_csv(combined_path)
    else:
        combined_df = pd.DataFrame()

    plot_variable = st.selectbox(
        "Select joint/angle for mean ± SD preview",
        options=selected_columns,
        format_func=get_display_label,
        key="kinematics_mean_sd_preview_variable",
    )

    show_individual_lines = st.checkbox(
        "Show individual trial lines",
        value=True,
        key="kinematics_show_individual_trial_lines",
    )

    plot_df = mean_sd_df[mean_sd_df["variable"] == plot_variable].copy()

    if not plot_df.empty:
        x = plot_df["percent_cycle"].to_numpy(dtype=float)
        mean = plot_df["mean_deg"].to_numpy(dtype=float)
        sd = plot_df["sd_deg"].to_numpy(dtype=float)

        fig_mean_sd = go.Figure()

        # Individual trial lines
        if show_individual_lines and not combined_df.empty:
            individual_df = combined_df[
                combined_df["variable"] == plot_variable
            ].copy()

            for trial_label, trial_df in individual_df.groupby("trial_label"):
                trial_df = trial_df.sort_values("percent_cycle")

                fig_mean_sd.add_trace(
                    go.Scatter(
                        x=trial_df["percent_cycle"],
                        y=trial_df["value_deg"],
                        mode="lines",
                        name=trial_label,
                        line=dict(width=1),
                        opacity=0.35,
                    )
                )

        # Upper SD boundary
        fig_mean_sd.add_trace(
            go.Scatter(
                x=x,
                y=mean + sd,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower SD boundary with fill
        fig_mean_sd.add_trace(
            go.Scatter(
                x=x,
                y=mean - sd,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="± SD",
            )
        )

        # Mean line
        fig_mean_sd.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name="Mean",
                line=dict(width=4),
            )
        )

        fig_mean_sd.update_layout(
            title=f"{get_display_label(plot_variable)} - Individual Trials and Mean ± SD",
            xaxis_title="Movement cycle (%)",
            yaxis_title="Angle (degrees)",
            height=600,
            autosize=True,
            font=dict(
                family="Poppins, Arial, Helvetica, sans-serif",
                size=12,
            ),
            margin=dict(l=70, r=30, t=80, b=130),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_mean_sd,
            use_container_width=True,
            config={"responsive": True},
        )

    else:
        st.warning("No mean ± SD data available for the selected joint.")

# -----------------------------------------------------------------------------
# Primary workflow status and navigation
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Primary Trial Workflow Status")
show_workflow_status(loaded_trials[0]["project_path"])

workflow_navigation(
    back_step="preprocessing",
    next_step="report",
    next_enabled=can_continue,
    next_disabled_message="Run kinematics for all selected trials before continuing.",
)