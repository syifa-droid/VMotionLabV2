from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.app_style import apply_global_style
from modules.project_manager import (
    file_exists,
    load_metadata,
    save_metadata,
)
from modules.state import initialize_state, require_active_project
from modules.trial_trimmer import (
    get_preview_columns,
    get_time_bounds,
    load_motion_raw,
    trim_trial_files,
)
from modules.ui_navigation import (
    show_project_header,
    show_workflow_status,
    workflow_navigation,
)


st.set_page_config(
    page_title="Trim Trial | VMotionLab",
    layout="wide",
)
apply_global_style()

initialize_state()
project_path = require_active_project()

st.title("Trim Trial")
st.caption("Select the useful movement window before preprocessing.")

show_project_header()

st.info(
    "Use this page to remove preparation time, long pauses, walking into position, "
    "turning around, or unnecessary movement before preprocessing."
)


st.warning(
    "Trimming should include the clinically relevant movement only. "
    "Poor trimming can affect interpolation, filtering, statistics, and report results."
)

st.warning(
    "Video recording_pose.mp4 is not available for preview. "
    "Open the video on project files and select time range manually"
    
)


st.divider()


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------

motion_raw_exists = file_exists(project_path, "motion_raw.csv")
recording_exists = file_exists(project_path, "recording.mp4")

if not motion_raw_exists:
    st.error("`motion_raw.csv` was not found. Please complete the Camera step first.")

    st.divider()
    show_workflow_status(project_path)

    workflow_navigation(
        back_step="upload_video",
        next_step="preprocessing",
        next_enabled=False,
        next_disabled_message="Camera output `motion_raw.csv` is required before trimming.",
    )

    st.stop()


# -----------------------------------------------------------------------------
# Load raw motion
# -----------------------------------------------------------------------------

try:
    motion_raw = load_motion_raw(project_path)
    time_min, time_max = get_time_bounds(motion_raw)

except Exception as exc:
    st.error(f"Could not load `motion_raw.csv`: {exc}")

    workflow_navigation(
        back_step="camera",
        next_step="preprocessing",
        next_enabled=False,
        next_disabled_message="Fix `motion_raw.csv` before continuing.",
    )

    st.stop()

if time_max <= time_min:
    st.error("The recording duration is invalid. Please record the trial again.")
    st.stop()

# -----------------------------------------------------------------------------
# Select trimming window
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Select Useful Movement Window")

st.write(
    "Move the start and end handles to keep only the useful movement period. "
    "The trimmed file will start again at time 0 seconds."
)

step = 0.01

selected_start, selected_end = st.slider(
    "Trim window in original recording time",
    min_value=float(time_min),
    max_value=float(time_max),
    value=(float(time_min), float(time_max)),
    step=step,
    format="%.2f s",
)

if selected_end <= selected_start:
    st.error("End time must be greater than start time.")
    st.stop()


selected_duration = selected_end - selected_start

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected start", f"{selected_start:.2f} s")

with col2:
    st.metric("Selected end", f"{selected_end:.2f} s")

with col3:
    st.metric("Selected duration", f"{selected_duration:.2f} s")


# -----------------------------------------------------------------------------
# Preview plot
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Motion Preview")

preview_columns = get_preview_columns(motion_raw)

if not preview_columns:
    st.warning("No numeric motion columns were found for preview plotting.")

else:
    default_index = 0

    if "knee_flexion_r" in preview_columns:
        default_index = preview_columns.index("knee_flexion_r")
    elif "knee_flexion_l" in preview_columns:
        default_index = preview_columns.index("knee_flexion_l")

    preview_column = st.selectbox(
        "Preview angle",
        preview_columns,
        index=default_index,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=motion_raw["time"],
            y=motion_raw[preview_column],
            mode="lines",
            name=preview_column,
        )
    )

    fig.add_vrect(
        x0=selected_start,
        x1=selected_end,
        opacity=0.2,
        line_width=0,
        annotation_text="Selected window",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"{preview_column} with selected trim window",
        xaxis_title="Time (s)",
        yaxis_title="Angle (degrees)",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Save trimmed files
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Save Trimmed Trial")

trim_landmarks = st.checkbox(
    "Also create landmarks_trimmed.csv",
    value=True,
    help="Recommended. This keeps landmark data synchronized with the trimmed motion window.",
)

save_button = st.button(
    "Save Trimmed Trial",
    type="primary",
    use_container_width=True,
)

if save_button:
    try:
        trim_summary = trim_trial_files(
            project_path=project_path,
            start_time=float(selected_start),
            end_time=float(selected_end),
            trim_landmarks=bool(trim_landmarks),
        )

        metadata = load_metadata(project_path)

        if metadata is not None:
            metadata["current_step"] = "trim_trial"
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "processing" not in metadata:
                metadata["processing"] = {}

            metadata["processing"]["trim"] = trim_summary

            if "files" not in metadata:
                metadata["files"] = {}

            metadata["files"]["motion_trimmed"] = "motion_trimmed.csv"

            if trim_summary.get("landmarks_trimmed_saved"):
                metadata["files"]["landmarks_trimmed"] = "landmarks_trimmed.csv"

            save_metadata(project_path, metadata)

        st.success("Trimmed trial saved successfully.")

        summary_df = pd.DataFrame(
            [
                {
                    "Output": "motion_trimmed.csv",
                    "Status": "Saved"
                    if file_exists(project_path, "motion_trimmed.csv")
                    else "Missing",
                },
                {
                    "Output": "landmarks_trimmed.csv",
                    "Status": "Saved"
                    if file_exists(project_path, "landmarks_trimmed.csv")
                    else "Not created",
                },
                {
                    "Output": "metadata.json",
                    "Status": "Updated",
                },
            ]
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write(
            f"**Original window:** {selected_start:.2f}–{selected_end:.2f} s  \n"
            f"**Trimmed duration:** {trim_summary['trimmed_duration_seconds']:.2f} s  \n"
            f"**Raw rows:** {trim_summary['raw_rows']}  \n"
            f"**Trimmed rows:** {trim_summary['trimmed_rows']}"
        )

        st.rerun()

    except Exception as exc:
        st.error(f"Could not save trimmed trial: {exc}")


# -----------------------------------------------------------------------------
# Output validation
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Trim Output Files")

if file_exists(project_path, "motion_trimmed.csv"):
    st.success("Trimmed motion data: `motion_trimmed.csv` found")
else:
    st.warning("Trimmed motion data: `motion_trimmed.csv` not found yet")

if file_exists(project_path, "landmarks_trimmed.csv"):
    st.success("Trimmed landmark data: `landmarks_trimmed.csv` found")
else:
    st.info("Trimmed landmark data: `landmarks_trimmed.csv` not found")


st.divider()
st.markdown("### Workflow Status")
show_workflow_status(project_path)

trim_complete = file_exists(project_path, "motion_trimmed.csv")

workflow_navigation(
    back_step="camera",
    next_step="preprocessing",
    next_enabled=trim_complete,
    next_disabled_message="Trim Trial is not complete yet. `motion_trimmed.csv` is required before continuing.",
)