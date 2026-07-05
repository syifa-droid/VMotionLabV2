from datetime import datetime
from pathlib import Path

import streamlit as st

from modules.app_style import apply_global_style
from modules.project_manager import (
    create_project,
    list_recent_projects,
    load_metadata,
    open_project_folder,
)
from modules.state import (
    initialize_state,
    set_active_project,
)
from modules.ui_navigation import workflow_navigation


st.set_page_config(
    page_title="New Analysis | VMotionLabV2",
    page_icon="🦿",
    layout="wide",
)

apply_global_style()
initialize_state()

st.title("New Analysis")
st.caption("Create or resume an analysis project.")


# -----------------------------------------------------------------------------
# Create new analysis
# -----------------------------------------------------------------------------

st.markdown("### Create New Analysis")

with st.form("new_analysis_form"):
    subject_id = st.text_input(
        "Subject / Participant ID",
        value="S001",
    )

    task = st.text_input(
        "Movement task",
        value="Walking",
    )

    trial_name = st.text_input(
        "Trial name",
        value="Trial_1",
    )

    side = st.selectbox(
        "Side",
        options=["Right", "Left", "Bilateral", "Not specified"],
        index=0,
    )

    clinician = st.text_input(
        "Data taker / clinician",
        value="",
    )

    notes = st.text_area(
        "Notes",
        value="",
    )

    submitted = st.form_submit_button(
        "Create Analysis",
        type="primary",
        use_container_width=True,
    )

if submitted:
    try:
        result = create_project(
            subject_id=subject_id,
            task=task,
            trial_name=trial_name,
            side=side,
            clinician=clinician,
            notes=notes,
        )

        if isinstance(result, dict):
            project_path = result.get("project_path")

            if project_path is None:
                project_path = result.get("path")

            if project_path is None:
                raise ValueError(
                    "create_project() returned metadata but no project_path was found."
                )
        else:
            project_path = result

        set_active_project(project_path)

        st.success("New analysis project created.")
        st.info(f"Active project: `{Path(project_path).name}`")

        st.switch_page("pages/2_Upload_Video.py")

    except Exception as exc:
        st.error(f"Could not create analysis project: {exc}")


# -----------------------------------------------------------------------------
# Resume recent analysis
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Resume Recent Analysis")

try:
    recent_projects = list_recent_projects(limit=10)

except Exception:
    recent_projects = []


if not recent_projects:
    st.info("No previous analysis projects found.")

else:
    for index, metadata in enumerate(recent_projects):
        project_path = metadata.get("project_path")

        if project_path is None:
            continue

        loaded_metadata = load_metadata(project_path) or metadata

        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(
                    f"**Subject:** {loaded_metadata.get('subject_id', '-')}"
                    f"  \n**Task:** {loaded_metadata.get('task', '-')}"
                    f"  \n**Trial:** {loaded_metadata.get('trial_name', '-')}"
                    f"  \n**Updated:** {loaded_metadata.get('updated_at', loaded_metadata.get('created_at', '-'))}"
                )

            with col2:
                if st.button(
                    "Resume",
                    key=f"resume_{index}",
                    use_container_width=True,
                ):
                    set_active_project(project_path)

                    current_step = loaded_metadata.get("current_step", "new_analysis")

                    if current_step == "camera":
                        current_step = "upload_video"

                    page_map = {
                        "new_analysis": "pages/1_New_Analysis.py",
                        "upload_video": "pages/2_Upload_Video.py",
                        "trim_trial": "pages/3_Trim_Trial.py",
                        "preprocessing": "pages/4_Preprocessing.py",
                        "kinematics": "pages/5_Kinematics.py",
                        "multi_trial": "pages/6_Multi_Trial.py",
                        "report": "pages/7_Report.py",
                    }

                    st.switch_page(
                        page_map.get(current_step, "pages/2_Upload_Video.py")
                    )

            with col3:
                if st.button(
                    "Open Folder",
                    key=f"open_folder_{index}",
                    use_container_width=True,
                ):
                    success = open_project_folder(project_path)

                    if success:
                        st.toast("Project folder opened.")
                    else:
                        st.error("Could not open folder.")


# -----------------------------------------------------------------------------
# Workflow navigation
# -----------------------------------------------------------------------------

st.divider()

st.info(
    "Create or resume an analysis first. Then continue to Upload Video."
)

workflow_navigation(
    back_step="home",
    next_step="upload_video",
    next_enabled=False,
    next_disabled_message="Create or resume an analysis first.",
)