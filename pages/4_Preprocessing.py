from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.app_style import apply_global_style
from modules.preprocessing import (
    MOTION_FILTERED_FILENAME,
    MOTION_RAW_FILENAME,
    PROCESSING_INFO_FILENAME,
    estimate_sampling_frequency,
    get_available_angle_columns,
    get_display_label,
    load_motion_file,
    preprocess_motion_file,
)
from modules.project_manager import (
    file_exists,
    load_metadata,
    save_metadata,
)
from modules.state import initialize_state, require_active_projects
from modules.ui_navigation import (
    show_workflow_status,
    workflow_navigation,
)


st.set_page_config(
    page_title="Preprocessing | VMotionLabV2",
    layout="wide",
)

apply_global_style()
initialize_state()

active_project_paths = require_active_projects()

st.title("Preprocessing")
st.caption(
    "Interpolate and filter selected clinical angle signals for one or more selected trials."
)

st.info(
    "Preprocessing will be applied to all selected active trials. "
    "Go back to Upload Videos if you want to change the selected trial(s)."
)


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

    raw_path = project_path / MOTION_RAW_FILENAME

    if not raw_path.exists():
        trial_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
                "motion_raw.csv": "Missing",
                "Rows": "",
                "Duration (s)": "",
                "Sampling frequency (Hz)": "",
            }
        )
        continue

    try:
        df = load_motion_file(raw_path)
        fs = estimate_sampling_frequency(df)
        duration = float(df["time"].max() - df["time"].min())
        angle_columns = get_available_angle_columns(df)

        loaded_trials.append(
            {
                "project_path": project_path,
                "metadata": metadata,
                "df": df,
                "sampling_frequency": fs,
                "duration": duration,
                "angle_columns": angle_columns,
            }
        )

        trial_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
                "motion_raw.csv": "Found",
                "Rows": len(df),
                "Duration (s)": round(duration, 2),
                "Sampling frequency (Hz)": round(fs, 2),
            }
        )

    except Exception as exc:
        trial_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
                "motion_raw.csv": f"Error: {exc}",
                "Rows": "",
                "Duration (s)": "",
                "Sampling frequency (Hz)": "",
            }
        )

st.dataframe(
    pd.DataFrame(trial_rows),
    use_container_width=True,
    hide_index=True,
)

if not loaded_trials:
    st.error("No selected trial has a valid motion_raw.csv file.")
    st.stop()


# -----------------------------------------------------------------------------
# Common angle columns
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Select Joints / Angles to Preprocess")

common_columns = set(loaded_trials[0]["angle_columns"])

for trial in loaded_trials[1:]:
    common_columns = common_columns.intersection(set(trial["angle_columns"]))

common_columns = list(common_columns)

preferred_order = loaded_trials[0]["angle_columns"]

common_columns = [
    column for column in preferred_order
    if column in common_columns
]

if not common_columns:
    st.error(
        "No common clinical angle columns were found across the selected trials. "
        "Please select trials processed with the same pose/angle configuration."
    )
    st.stop()

selected_columns = st.multiselect(
    "Clinical angle variables to preprocess",
    options=common_columns,
    default=common_columns,
    format_func=get_display_label,
    key="preprocessing_selected_columns_multi_trial",
)

if not selected_columns:
    st.warning("Select at least one joint/angle column to preprocess.")
    st.stop()

with st.expander("Selected joints / angles", expanded=False):
    selected_table = pd.DataFrame(
        {
            "Column": selected_columns,
            "Display label": [get_display_label(col) for col in selected_columns],
        }
    )

    st.dataframe(
        selected_table,
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Preview primary trial signal
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Signal Preview Before Preprocessing")

primary_trial = loaded_trials[0]
primary_df = primary_trial["df"]
preview_columns = selected_columns[:6]

st.caption(
    "Preview shows the first selected active trial only. "
    "Preprocessing will still be applied to all selected trials."
)

fig_preview = go.Figure()

for column in preview_columns:
    fig_preview.add_trace(
        go.Scatter(
            x=primary_df["time"],
            y=pd.to_numeric(primary_df[column], errors="coerce"),
            mode="lines",
            name=get_display_label(column),
        )
    )

fig_preview.update_layout(
    title="Selected Clinical Angle Signals - Primary Trial",
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
    fig_preview,
    use_container_width=True,
    config={"responsive": True},
)

if len(selected_columns) > 6:
    st.caption(
        f"Preview shows first 6 selected signals. Total selected: {len(selected_columns)}."
    )


# -----------------------------------------------------------------------------
# Preprocessing settings
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Preprocessing Settings")

apply_interpolation = st.checkbox(
    "Interpolate missing values",
    value=True,
    key="preprocessing_apply_interpolation",
)

apply_filter = st.checkbox(
    "Apply Butterworth low-pass filter",
    value=True,
    key="preprocessing_apply_filter",
)

sampling_frequencies = [
    trial["sampling_frequency"]
    for trial in loaded_trials
    if trial["sampling_frequency"] > 0
]

if not sampling_frequencies:
    st.error("Could not estimate sampling frequency from the selected trials.")
    st.stop()

minimum_sampling_frequency = min(sampling_frequencies)
minimum_nyquist = minimum_sampling_frequency / 2.0

col_cutoff, col_order = st.columns(2)

with col_cutoff:
    cutoff_frequency = st.number_input(
        "Cutoff frequency (Hz)",
        min_value=0.1,
        max_value=20.0,
        value=min(6.0, max(0.1, minimum_nyquist - 0.1)),
        step=0.5,
        key="preprocessing_cutoff_frequency",
    )

with col_order:
    filter_order = st.number_input(
        "Filter order",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        key="preprocessing_filter_order",
    )

if apply_filter:
    st.caption(
        f"Lowest sampling frequency among selected trials: {minimum_sampling_frequency:.2f} Hz | "
        f"Lowest Nyquist frequency: {minimum_nyquist:.2f} Hz"
    )

    if cutoff_frequency >= minimum_nyquist:
        st.error(
            f"Cutoff frequency must be lower than the lowest Nyquist frequency across selected trials. "
            f"Current cutoff: {cutoff_frequency:.2f} Hz, lowest Nyquist: {minimum_nyquist:.2f} Hz."
        )
        st.stop()


# -----------------------------------------------------------------------------
# Run preprocessing for all selected trials
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Run Preprocessing")

if st.button(
    "Run Preprocessing for Selected Trial(s)",
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

        input_path = project_path / MOTION_RAW_FILENAME
        output_path = project_path / MOTION_FILTERED_FILENAME
        info_path = project_path / PROCESSING_INFO_FILENAME

        status_box.info(
            f"Preprocessing {index + 1}/{total_trials}: `{project_path.name}`"
        )

        try:
            info = preprocess_motion_file(
                input_path=input_path,
                output_path=output_path,
                info_path=info_path,
                selected_columns=selected_columns,
                apply_interpolation=apply_interpolation,
                apply_filter=apply_filter,
                cutoff_frequency=cutoff_frequency,
                filter_order=int(filter_order),
            )

            metadata = load_metadata(project_path) or metadata

            metadata["current_step"] = "preprocessing"
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "processing" not in metadata:
                metadata["processing"] = {}

            metadata["processing"]["preprocessing"] = info

            if "files" not in metadata:
                metadata["files"] = {}

            metadata["files"]["motion_filtered"] = MOTION_FILTERED_FILENAME
            metadata["files"]["processing_info"] = PROCESSING_INFO_FILENAME

            save_metadata(project_path, metadata)

            results.append(
                {
                    "Project": project_path.name,
                    "Status": "Completed",
                    "Filtered file": MOTION_FILTERED_FILENAME,
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

    status_box.success("Preprocessing finished.")

    if results:
        st.success(f"{len(results)} trial(s) preprocessed successfully.")
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True,
        )

    if failures:
        st.error(f"{len(failures)} trial(s) failed.")
        st.dataframe(
            pd.DataFrame(failures),
            use_container_width=True,
            hide_index=True,
        )

    st.rerun()


# -----------------------------------------------------------------------------
# Output validation
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Preprocessing Output Status")

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
            "motion_filtered.csv": "Yes" if file_exists(project_path, MOTION_FILTERED_FILENAME) else "No",
            "processing_info.json": "Yes" if file_exists(project_path, PROCESSING_INFO_FILENAME) else "No",
        }
    )

st.dataframe(
    pd.DataFrame(output_rows),
    use_container_width=True,
    hide_index=True,
)

can_continue = all(
    file_exists(trial["project_path"], MOTION_FILTERED_FILENAME)
    for trial in loaded_trials
)


# -----------------------------------------------------------------------------
# Primary workflow status and navigation
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Primary Trial Workflow Status")
show_workflow_status(loaded_trials[0]["project_path"])

workflow_navigation(
    back_step="upload_video",
    next_step="kinematics",
    next_enabled=can_continue,
    next_disabled_message="Run preprocessing for all selected trials before continuing.",
)