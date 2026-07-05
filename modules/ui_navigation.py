from pathlib import Path

import streamlit as st

from modules.project_manager import file_exists, load_metadata


PAGES = {
    "home": "Home.py",
    "upload_video": "pages/2_Upload_Video.py",
    "preprocessing": "pages/4_Preprocessing.py",
    "kinematics": "pages/5_Kinematics.py",
    "report": "pages/7_Report.py",
}


WORKFLOW_ORDER = [
    "upload_video",
    "preprocessing",
    "kinematics",
    "report",
]

STEP_LABELS = {
    "upload_video": "Upload Videos",
    "preprocessing": "Preprocessing",
    "kinematics": "Kinematics",
    "report": "Report",
}


STEP_FILES = {
    "upload_video": ["motion_raw.csv"],
    "preprocessing": ["motion_filtered.csv"],
    "kinematics": ["statistics.csv", "normalized_curves.csv"],
    "report": ["report/report.pdf"],
}


def get_step_label(step: str) -> str:
    return STEP_LABELS.get(step, step.replace("_", " ").title())


def get_page(step: str) -> str:
    return PAGES.get(step, "Home.py")


def switch_to_step(step: str) -> None:
    st.switch_page(get_page(step))


def is_step_complete(project_path: Path | str, step: str) -> bool:
    project_path = Path(project_path)

    required_files = STEP_FILES.get(step, [])

    if not required_files:
        return False

    return all(file_exists(project_path, filename) for filename in required_files)


def show_project_header() -> None:
    active_project_paths = st.session_state.get("active_project_paths", [])

    if not active_project_paths:
        active_project_path = st.session_state.get("active_project_path")

        if active_project_path:
            active_project_paths = [active_project_path]

    if not active_project_paths:
        st.info("No trial selected.")
        return

    if len(active_project_paths) == 1:
        project_path = Path(active_project_paths[0])
        metadata = load_metadata(project_path)

        if metadata is None:
            st.info(f"Selected trial: `{project_path.name}`")
            return

        subject_id = metadata.get("subject_id", "Unknown")
        task = metadata.get("task", "Unknown")
        trial_name = metadata.get("trial_name", project_path.name)

        st.info(
            f"Selected trial: **{subject_id}** | **{task}** | **{trial_name}**"
        )

    else:
        st.info(f"Selected trials: **{len(active_project_paths)} trials**")

        rows = []

        for path in active_project_paths:
            project_path = Path(path)
            metadata = load_metadata(project_path) or {}

            rows.append(
                {
                    "Participant": metadata.get("subject_id", "Unknown"),
                    "Movement / Task": metadata.get("task", "Unknown"),
                    "Trial": metadata.get("trial_name", project_path.name),
                    "Project": project_path.name,
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


def show_workflow_status(project_path: Path | str) -> None:
    project_path = Path(project_path)

    cols = st.columns(len(WORKFLOW_ORDER))

    for col, step in zip(cols, WORKFLOW_ORDER):
        complete = is_step_complete(project_path, step)
        label = get_step_label(step)

        with col:
            if complete:
                st.success(f"✅ {label}")
            else:
                st.warning(f"○ {label}")


def workflow_navigation(
    back_step: str | None = None,
    next_step: str | None = None,
    next_enabled: bool = True,
    next_disabled_message: str = "Complete the current step before continuing.",
) -> None:
    col_back, col_next = st.columns(2)

    with col_back:
        if back_step is not None:
            if st.button(
                f"⬅ Back to {get_step_label(back_step)}",
                use_container_width=True,
            ):
                switch_to_step(back_step)

    with col_next:
        if next_step is not None:
            if next_enabled:
                if st.button(
                    f"Continue to {get_step_label(next_step)} ➜",
                    type="primary",
                    use_container_width=True,
                ):
                    switch_to_step(next_step)
            else:
                st.button(
                    f"Continue to {get_step_label(next_step)} ➜",
                    disabled=True,
                    use_container_width=True,
                )
                st.caption(next_disabled_message)