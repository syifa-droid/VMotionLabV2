from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.app_style import apply_global_style
from modules.clinical_notes import (
    show_angle_convention_note,
    show_angle_limitation_note,
)
from modules.multi_trial import (
    MULTI_TRIAL_COMBINED_FILENAME,
    MULTI_TRIAL_INFO_FILENAME,
    MULTI_TRIAL_MEAN_SD_FILENAME,
    MULTI_TRIAL_STATISTICS_FILENAME,
    find_projects_with_normalized_curves,
    get_common_angle_columns,
    get_display_label,
    load_normalized_curves,
    make_project_label,
    run_multi_trial_analysis,
)
from modules.project_manager import (
    file_exists,
    load_metadata,
    save_metadata,
)
from modules.state import initialize_state, require_active_project
from modules.ui_navigation import (
    show_project_header,
    workflow_navigation,
)


st.set_page_config(
    page_title="Multi-Trial Analysis | VMotionLab",
    layout="wide",
)
apply_global_style()

initialize_state()
active_project_path = require_active_project()

st.title("Multi-Trial Analysis")
st.caption("Compare normalized curves across multiple trials using mean ± SD.")

show_project_header()

show_angle_convention_note()
show_angle_limitation_note()

st.info(
    "Multi-trial analysis uses `normalized_curves.csv` from completed trials. "
    "Each trial is already normalized to 0-100% movement cycle, allowing trials "
    "with different durations to be compared."
)

st.warning(
    "Select trials that represent the same movement task and similar recording setup. "
    "Combining different tasks, poor trimming, or inconsistent camera views can produce misleading averages."
)

st.divider()


# -----------------------------------------------------------------------------
# Find eligible projects
# -----------------------------------------------------------------------------

st.markdown("### Select Completed Trials")

eligible_projects = find_projects_with_normalized_curves(limit=50)

# -----------------------------------------------------------------------------
# Movement / task filter
# -----------------------------------------------------------------------------

available_tasks = sorted(
    list(
        {
            metadata.get("task", "Unknown")
            for metadata in eligible_projects
        }
    )
)

selected_task = st.selectbox(
    "Select movement / task",
    options=available_tasks,
)

eligible_projects = [
    metadata
    for metadata in eligible_projects
    if metadata.get("task", "Unknown") == selected_task
]


label_to_project = {
    make_project_label(metadata): metadata
    for metadata in eligible_projects
}

labels = list(label_to_project.keys())

default_labels = labels[: min(3, len(labels))]

selected_labels = st.multiselect(
    "Select trials/projects",
    options=labels,
    default=default_labels,
)

selected_project_paths = [
    Path(label_to_project[label]["project_path"])
    for label in selected_labels
]


if len(selected_project_paths) < 2:
    st.warning("Select at least two trials to continue.")
    st.stop()


selected_projects_table = []

for label in selected_labels:
    metadata = label_to_project[label]
    selected_projects_table.append(
        {
            "Subject": metadata.get("subject_id", ""),
            "Task": metadata.get("task", ""),
            "Trial": metadata.get("trial_name", ""),
            "Created": metadata.get("created_at", ""),
            "Project": metadata.get("project_name", ""),
        }
    )

st.dataframe(
    pd.DataFrame(selected_projects_table),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------------------------------------------------------
# Select variables
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Select Angle Variables")

try:
    common_angle_columns = get_common_angle_columns(selected_project_paths)
except Exception as exc:
    st.error(f"Could not inspect selected projects: {exc}")
    st.stop()


if not common_angle_columns:
    st.error(
        "No common angle variables were found across the selected trials. "
        "Make sure all selected projects were processed using the same VMotionLab version."
    )
    st.stop()


default_variables = []

for preferred in [
    "knee_flexion_r",
    "knee_flexion_l",
    "hip_flexion_r",
    "hip_flexion_l",
    "ankle_angle_r",
    "ankle_angle_l",
]:
    if preferred in common_angle_columns:
        default_variables.append(preferred)

if not default_variables:
    default_variables = common_angle_columns[: min(3, len(common_angle_columns))]

selected_variables = st.multiselect(
    "Select joint / clinical angle",
    options=common_angle_columns,
    default=default_variables,
    format_func=get_display_label,
)

if not selected_variables:
    st.warning("Select at least one angle variable.")
    st.stop()


primary_variable = st.selectbox(
    "Primary variable for mean ± SD graph",
    options=selected_variables,
    index=0,
    format_func=get_display_label,
)

show_individual_trials = st.checkbox(
    "Show individual trial lines in saved figure",
    value=True,
)


# -----------------------------------------------------------------------------
# Build preview mean ± SD
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Preview Mean ± SD")

try:
    preview_rows = []

    for project_path in selected_project_paths:
        normalized_df = load_normalized_curves(project_path)
        metadata = load_metadata(project_path) or {}

        temp = pd.DataFrame(
            {
                "project_name": project_path.name,
                "trial_name": metadata.get("trial_name", project_path.name),
                "percent_movement": normalized_df["percent_movement"],
                "angle_deg": pd.to_numeric(
                    normalized_df[primary_variable],
                    errors="coerce",
                ),
            }
        )

        preview_rows.append(temp)

    preview_df = pd.concat(preview_rows, ignore_index=True)

    mean_sd_preview = (
        preview_df
        .groupby("percent_movement", as_index=False)
        .agg(
            mean_deg=("angle_deg", "mean"),
            sd_deg=("angle_deg", "std"),
            n_trials=("angle_deg", "count"),
        )
    )

    mean_sd_preview["sd_deg"] = mean_sd_preview["sd_deg"].fillna(0)
    mean_sd_preview["upper"] = mean_sd_preview["mean_deg"] + mean_sd_preview["sd_deg"]
    mean_sd_preview["lower"] = mean_sd_preview["mean_deg"] - mean_sd_preview["sd_deg"]

    fig = go.Figure()

    for trial_name, trial_df in preview_df.groupby("trial_name"):
        fig.add_trace(
            go.Scatter(
                x=trial_df["percent_movement"],
                y=trial_df["angle_deg"],
                mode="lines",
                name=f"Trial: {trial_name}",
                opacity=0.35,
                line=dict(width=1),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=mean_sd_preview["percent_movement"],
            y=mean_sd_preview["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mean_sd_preview["percent_movement"],
            y=mean_sd_preview["lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="±1 SD",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mean_sd_preview["percent_movement"],
            y=mean_sd_preview["mean_deg"],
            mode="lines",
            name="Mean",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=f"{get_display_label(primary_variable)} - Mean ± SD",
        xaxis_title="Movement cycle (%)",
        yaxis_title="Angle (degrees)",
        height=650,
        autosize=True,
        margin=dict(l=70, r=30, t=80, b=150),
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
        config={
            "responsive": True,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "vmotionlab_multi_trial_mean_sd",
                "height": 900,
                "width": 1400,
                "scale": 2,
            },
        },
    )

except Exception as exc:
    st.error(f"Could not generate preview: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Run Multi-Trial Analysis")

st.write(
    "Outputs will be saved to the currently active project folder. "
    "Use the active project as the summary container for this multi-trial analysis."
)

run_button = st.button(
    "Generate Multi-Trial Mean ± SD",
    type="primary",
    use_container_width=True,
)

if run_button:
    try:
        summary = run_multi_trial_analysis(
            output_project_path=active_project_path,
            selected_project_paths=selected_project_paths,
            selected_variables=selected_variables,
            primary_variable=primary_variable,
            show_individual_trials=show_individual_trials,
        )

        metadata = load_metadata(active_project_path)

        if metadata is not None:
            metadata["current_step"] = "multi_trial"
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "processing" not in metadata:
                metadata["processing"] = {}

            metadata["processing"]["multi_trial"] = summary

            if "files" not in metadata:
                metadata["files"] = {}

            metadata["files"]["multi_trial_combined_curves"] = MULTI_TRIAL_COMBINED_FILENAME
            metadata["files"]["multi_trial_mean_sd"] = MULTI_TRIAL_MEAN_SD_FILENAME
            metadata["files"]["multi_trial_statistics"] = MULTI_TRIAL_STATISTICS_FILENAME
            metadata["files"]["multi_trial_info"] = MULTI_TRIAL_INFO_FILENAME

            save_metadata(active_project_path, metadata)

        st.success("Multi-trial analysis completed successfully.")
        st.rerun()

    except Exception as exc:
        st.error(f"Multi-trial analysis failed: {exc}")


# -----------------------------------------------------------------------------
# Output validation and display
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Multi-Trial Output Files")

output_files = {
    "Combined trial curves": MULTI_TRIAL_COMBINED_FILENAME,
    "Mean ± SD curves": MULTI_TRIAL_MEAN_SD_FILENAME,
    "Multi-trial statistics": MULTI_TRIAL_STATISTICS_FILENAME,
    "Multi-trial info": MULTI_TRIAL_INFO_FILENAME,
}

for label, filename in output_files.items():
    if file_exists(active_project_path, filename):
        st.success(f"{label}: `{filename}` found")
    else:
        st.warning(f"{label}: `{filename}` not found yet")


if file_exists(active_project_path, MULTI_TRIAL_STATISTICS_FILENAME):
    st.divider()
    st.markdown("### Multi-Trial Statistics")

    try:
        stats_df = pd.read_csv(active_project_path / MULTI_TRIAL_STATISTICS_FILENAME)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.warning(f"Could not load multi-trial statistics: {exc}")


if file_exists(active_project_path, MULTI_TRIAL_MEAN_SD_FILENAME):
    st.divider()
    st.markdown("### Saved Mean ± SD Data Preview")

    try:
        mean_sd_df = pd.read_csv(active_project_path / MULTI_TRIAL_MEAN_SD_FILENAME)
        st.dataframe(mean_sd_df.head(20), use_container_width=True, hide_index=True)

    except Exception as exc:
        st.warning(f"Could not load multi-trial mean ± SD data: {exc}")

st.divider()
st.markdown("### Continue Workflow")

multi_trial_completed = (
    file_exists(active_project_path, MULTI_TRIAL_MEAN_SD_FILENAME)
    and file_exists(active_project_path, MULTI_TRIAL_STATISTICS_FILENAME)
)

if multi_trial_completed:
    st.success("Multi-Trial Analysis is complete. The report will include multi-trial results.")
else:
    st.info(
        "Multi-Trial Analysis is optional. You can generate it first, or continue to Report "
        "to create a single-trial report."
    )

workflow_navigation(
    back_step="kinematics",
    next_step="report",
    next_enabled=True,
)