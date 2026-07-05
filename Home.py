from pathlib import Path

import streamlit as st

from modules.app_style import apply_global_style
from modules.project_manager import (
    get_current_projects,
    list_project_metadata,
    load_metadata,
    set_current_projects,
)
from modules.state import initialize_state


st.set_page_config(
    page_title="VMotionLabV2",
    layout="wide",
)

apply_global_style()
initialize_state()

st.image(
    "assets/logo.png",
    use_container_width=True
)

st.set_page_config(
    page_title="VMotionLab",
    layout="wide",
)
apply_global_style()

initialize_state()

st.caption(
    "Author: Syifa Fauziah, B.PO, M.MedicalEng"
)



st.markdown(
    """
    <div class="vmotion-title">VMotionLabV2</div>
    <div class="vmotion-subtitle">Markerless Video-Based Motion Analysis (for research and education)</div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Upload one or more already-trimmed movement videos, process pose data, "
    "preprocess joint angles, analyze kinematics, compare selected trials, and generate reports."
)


# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Workflow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success("1. Upload Videos")

with col2:
    st.info("2. Preprocessing")

with col3:
    st.info("3. Kinematics / Comparison")

with col4:
    st.info("4. Report")


st.divider()
st.markdown("### Start")

if st.button("Start / Upload Videos", type="primary", use_container_width=True):
    st.switch_page("pages/2_Upload_Video.py")


# -----------------------------------------------------------------------------
# Selected trial(s)
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Selected Trial(s)")

selected_project_paths = get_current_projects()

if selected_project_paths:
    st.success(f"{len(selected_project_paths)} trial(s) selected.")

    selected_rows = []

    for project_path in selected_project_paths:
        project_path = Path(project_path)
        metadata = load_metadata(project_path) or {}

        selected_rows.append(
            {
                "Participant": metadata.get("subject_id", "Unknown"),
                "Movement / Task": metadata.get("task", "Unknown"),
                "Trial": metadata.get("trial_name", project_path.name),
                "Project": project_path.name,
            }
        )

    st.dataframe(
        selected_rows,
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Go to Preprocessing", use_container_width=True):
            st.switch_page("pages/4_Preprocessing.py")

    with col_b:
        if st.button("Go to Kinematics / Comparison", use_container_width=True):
            st.switch_page("pages/5_Kinematics.py")

    with col_c:
        if st.button("Go to Report", use_container_width=True):
            st.switch_page("pages/7_Report.py")

else:
    st.warning("No trial selected yet. Upload and process video(s) first.")


# -----------------------------------------------------------------------------
# Existing processed trials
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Existing Processed Trials")

projects = list_project_metadata()

processed_projects = [
    project
    for project in projects
    if Path(project["project_path"]).joinpath("motion_raw.csv").exists()
]

if not processed_projects:
    st.info("No processed trials found yet.")

else:
    labels = []

    for project in processed_projects:
        subject_id = project.get("subject_id", "Unknown")
        task = project.get("task", "Unknown")
        trial_name = project.get("trial_name", "Trial")
        project_name = project.get("project_name", "")

        labels.append(f"{subject_id} | {task} | {trial_name} | {project_name}")

    current_paths = [str(path) for path in selected_project_paths]

    default_labels = []

    for label, project in zip(labels, processed_projects):
        if str(project.get("project_path")) in current_paths:
            default_labels.append(label)

    selected_labels = st.multiselect(
        "Select processed trial(s)",
        options=labels,
        default=default_labels,
        key="home_selected_trials_multiselect",
    )

    if st.button("Set Selected Trial(s)", use_container_width=True):
        selected_paths = []

        for label, project in zip(labels, processed_projects):
            if label in selected_labels:
                selected_paths.append(Path(project["project_path"]))

        if selected_paths:
            set_current_projects(selected_paths)
            st.session_state["active_project_path"] = str(selected_paths[0])
            st.session_state["active_project_paths"] = [
                str(path) for path in selected_paths
            ]
            st.success(f"{len(selected_paths)} trial(s) selected.")
            st.rerun()
        else:
            st.warning("Please select at least one trial.")


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Notes")

st.markdown(
    """
    - VMotionLabV2 uses uploaded trimmed videos only.
    - Each uploaded video becomes one trial.
    - Videos can belong to the same participant, different participants, or different trials.
    - Kinematics / Comparison supports one or more selected trials.
    - Multi-trial comparison is now integrated into the Kinematics page.
    """
)