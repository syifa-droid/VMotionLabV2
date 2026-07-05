from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.app_style import apply_global_style
from modules.project_manager import (
    build_project_folder_name,
    create_project,
    file_exists,
    find_completed_projects,
    get_current_projects,
    get_project_path,
    load_metadata,
    save_metadata,
    set_current_project,
    set_current_projects,
)
from modules.state import initialize_state
from modules.ui_navigation import (
    show_workflow_status,
    workflow_navigation,
)
from modules.video_processor import process_uploaded_video


st.set_page_config(
    page_title="Upload Videos | VMotionLabV2",
    layout="wide",
)

apply_global_style()
initialize_state()

st.title("Upload Videos")
st.caption(
    "Upload one or more already-trimmed movement videos and process them offline using markerless pose estimation."
)

st.warning(
    "Please upload videos that have already been trimmed to the movement task or cycle you want to analyze. "
    "Example: one gait cycle, one sit-to-stand repetition, one squat repetition, or one upper-limb movement trial."
)


# -----------------------------------------------------------------------------
# Good video recording guide
# -----------------------------------------------------------------------------

st.markdown("### Good Video Recording Guide")

with st.expander("How to record a good video for VMotionLabV2", expanded=True):
    st.markdown(
        """
        For best motion analysis results, use a video that follows these recommendations:

        **Camera position**
        - Place the camera perpendicular to the movement direction.
        - For walking, record from the side view if sagittal-plane analysis is intended.
        - Keep the camera stable using a tripod or fixed surface.
        - Avoid moving or rotating the camera during recording.

        **Participant position**
        - The full body or relevant body region should be visible throughout the movement.
        - Important landmarks should not be blocked by clothing, objects, or other people.
        - Use clothing that contrasts with the background.
        - Avoid loose clothing that hides the hip, knee, ankle, shoulder, elbow, or wrist.

        **Video quality**
        - Use good lighting.
        - Avoid backlight, shadows, and very dark environments.
        - Use landscape orientation when possible.
        - Use a clear video with minimal blur.
        - A frame rate of 30 fps is recommended.

        **Movement**
        - The movement should stay mostly in one plane.
        - For sagittal analysis, avoid large out-of-plane rotation.
        - Record one clear task per video.

        **Clinical note**
        - VMotionLabV2 estimates 2D markerless clinical angles from video.
        - Result quality depends on video quality, camera placement, and landmark visibility.
        - Final validation and interpretation remain the responsibility of the data taker or clinician.
        """
    )


# -----------------------------------------------------------------------------
# Upload multiple videos
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Upload Trimmed Video Files")

uploaded_files = st.file_uploader(
    "Upload one or more trimmed video files",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True,
    key="upload_multiple_trimmed_videos",
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} video file(s) selected.")
else:
    st.info("Upload one or more trimmed videos to begin.")
    uploaded_files = []


# -----------------------------------------------------------------------------
# Metadata table for each uploaded video
# -----------------------------------------------------------------------------

if uploaded_files:
    st.divider()
    st.markdown("### Assign Participant, Movement, and Trial")

    st.caption(
        "Each uploaded video will become one separate trial folder. "
        "Videos may belong to the same participant, different participants, or different trials."
    )

    st.markdown("#### Default values")

    col_default_1, col_default_2, col_default_3 = st.columns(3)

    with col_default_1:
        default_participant_id = st.text_input(
            "Default participant ID",
            value="P001",
            key="upload_default_participant_id",
        )

    with col_default_2:
        default_movement_task = st.text_input(
            "Default movement / task",
            value="Walking",
            key="upload_default_movement_task",
        )

    with col_default_3:
        default_side = st.selectbox(
            "Default side",
            options=["", "Right", "Left", "Bilateral", "Not applicable"],
            index=0,
            key="upload_default_side",
        )

    default_rows = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        default_rows.append(
            {
                "file_name": uploaded_file.name,
                "participant_id": default_participant_id,
                "movement_task": default_movement_task,
                "trial_name": f"Trial{index:02d}",
                "side": default_side,
                "notes": "",
            }
        )

    signature = "|".join(
        [file.name for file in uploaded_files]
        + [default_participant_id, default_movement_task, default_side]
    )

    editor_key = hashlib.md5(signature.encode("utf-8")).hexdigest()

    metadata_df = pd.DataFrame(default_rows)

    edited_df = st.data_editor(
        metadata_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["file_name"],
        key=f"upload_video_metadata_editor_{editor_key}",
        column_config={
            "file_name": st.column_config.TextColumn("Video file"),
            "participant_id": st.column_config.TextColumn("Participant ID"),
            "movement_task": st.column_config.TextColumn("Movement / Task"),
            "trial_name": st.column_config.TextColumn("Trial name"),
            "side": st.column_config.TextColumn("Side"),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )

    overwrite_existing = st.checkbox(
        "Overwrite existing trial folders if participant / movement / trial names match",
        value=False,
        key="upload_video_overwrite_existing",
    )

    save_annotated_video = st.checkbox(
        "Save annotated video with pose overlay",
        value=False,
        key="upload_video_save_annotated",
    )

    target_names = []

    for _, row in edited_df.iterrows():
        target_names.append(
            build_project_folder_name(
                subject_id=row["participant_id"],
                task=row["movement_task"],
                trial_name=row["trial_name"],
            )
        )

    duplicate_names = sorted(
        {
            name for name in target_names
            if target_names.count(name) > 1
        }
    )

    if duplicate_names:
        st.error(
            "Duplicate participant / movement / trial combinations found. "
            "Please make each trial name unique."
        )

        for name in duplicate_names:
            st.code(name)

        st.stop()


# -----------------------------------------------------------------------------
# Process uploaded videos
# -----------------------------------------------------------------------------

if uploaded_files:
    st.divider()
    st.markdown("### Process Uploaded Videos")

    process_button = st.button(
        "Process All Uploaded Videos",
        type="primary",
        use_container_width=True,
        key="upload_video_process_all",
    )

    if process_button:
        file_lookup = {
            uploaded_file.name: uploaded_file
            for uploaded_file in uploaded_files
        }

        processed_projects = []
        failed_projects = []

        overall_progress = st.progress(0.0)
        status_box = st.empty()

        total_files = len(edited_df)

        for file_index, (_, row) in enumerate(edited_df.iterrows()):
            file_name = str(row["file_name"]).strip()
            uploaded_file = file_lookup[file_name]

            participant_id = str(row["participant_id"]).strip() or "Unknown"
            movement_task = str(row["movement_task"]).strip() or "Unknown"
            trial_name = str(row["trial_name"]).strip() or Path(file_name).stem
            side = str(row["side"]).strip()
            notes = str(row["notes"]).strip()

            project_path = get_project_path(
                subject_id=participant_id,
                task=movement_task,
                trial_name=trial_name,
            )

            if (
                project_path.exists()
                and (project_path / "motion_raw.csv").exists()
                and not overwrite_existing
            ):
                failed_projects.append(
                    {
                        "file": file_name,
                        "reason": (
                            "Trial folder already exists. Enable overwrite if you want to reprocess it."
                        ),
                    }
                )
                continue

            status_box.info(
                f"Processing {file_index + 1}/{total_files}: `{file_name}`"
            )

            try:
                project_result = create_project(
                    subject_id=participant_id,
                    task=movement_task,
                    trial_name=trial_name,
                    side=side,
                    clinician="",
                    notes=notes,
                    source_video_name=file_name,
                )

                project_path = Path(project_result["project_path"])

                suffix = Path(file_name).suffix.lower()

                if not suffix:
                    suffix = ".mp4"

                uploaded_video_path = project_path / f"uploaded_video{suffix}"

                video_bytes = uploaded_file.getbuffer().tobytes()
                uploaded_video_path.write_bytes(video_bytes)

                metadata = load_metadata(project_path)

                if metadata is not None:
                    metadata["current_step"] = "upload_video"
                    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

                    metadata["video"] = {
                        "source_video_name": file_name,
                        "uploaded_video": uploaded_video_path.name,
                    }

                    metadata["files"]["uploaded_video"] = uploaded_video_path.name

                    save_metadata(project_path, metadata)

                def update_progress(progress_value: float) -> None:
                    overall_value = (file_index + progress_value) / total_files
                    overall_progress.progress(min(float(overall_value), 1.0))

                summary = process_uploaded_video(
                    video_path=uploaded_video_path,
                    output_project_path=project_path,
                    save_annotated_video=save_annotated_video,
                    progress_callback=update_progress,
                )

                metadata = load_metadata(project_path)

                if metadata is not None:
                    metadata["current_step"] = "upload_video"
                    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

                    metadata["video_processing"] = summary

                    metadata["files"]["motion_raw"] = "motion_raw.csv"

                    if (project_path / "landmarks.csv").exists():
                        metadata["files"]["landmarks"] = "landmarks.csv"

                    if save_annotated_video and (project_path / "recording_with_pose.mp4").exists():
                        metadata["files"]["annotated_video"] = "recording_with_pose.mp4"

                    save_metadata(project_path, metadata)

                processed_projects.append(
                    {
                        "file": file_name,
                        "participant_id": participant_id,
                        "movement_task": movement_task,
                        "trial_name": trial_name,
                        "project_path": str(project_path),
                        "status": "Processed",
                    }
                )

            except Exception as exc:
                failed_projects.append(
                    {
                        "file": file_name,
                        "reason": str(exc),
                    }
                )

        overall_progress.progress(1.0)
        status_box.success("Batch processing finished.")

        if processed_projects:
            processed_project_paths = [
                Path(project["project_path"])
                for project in processed_projects
            ]

            set_current_projects(processed_project_paths)

            st.session_state["active_project_path"] = str(processed_project_paths[0])
            st.session_state["active_project_paths"] = [
                str(path) for path in processed_project_paths
            ]

            st.success(f"{len(processed_projects)} video(s) processed successfully.")

            st.dataframe(
                pd.DataFrame(processed_projects),
                use_container_width=True,
                hide_index=True,
            )

        if failed_projects:
            st.error(f"{len(failed_projects)} video(s) were not processed.")

            st.dataframe(
                pd.DataFrame(failed_projects),
                use_container_width=True,
                hide_index=True,
            )

        st.rerun()

# -----------------------------------------------------------------------------
# Select active processed trials
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Select Active Trial(s)")

completed_projects = find_completed_projects(
    required_files=["motion_raw.csv"]
)

if not completed_projects:
    st.info("No processed trials found yet.")
    can_continue = False

else:
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

    selected_labels = st.multiselect(
        "Choose one or more trials for preprocessing / kinematics / comparison",
        options=labels,
        default=default_labels,
        key="upload_video_active_trials_selector",
    )

    if not selected_labels:
        st.warning("Select at least one active trial.")
        can_continue = False

    else:
        selected_project_paths = [
            Path(label_to_metadata[label]["project_path"])
            for label in selected_labels
        ]

        set_current_projects(selected_project_paths)

        st.session_state["active_project_path"] = str(selected_project_paths[0])
        st.session_state["active_project_paths"] = [
            str(path) for path in selected_project_paths
        ]

        st.success(
            f"{len(selected_project_paths)} active trial(s) selected. "
            "The first selected trial will be used as the primary trial."
        )

        selected_table = []

        for label in selected_labels:
            metadata = label_to_metadata[label]
            project_path = Path(metadata["project_path"])

            selected_table.append(
                {
                    "Participant": metadata.get("subject_id", "Unknown"),
                    "Movement / Task": metadata.get("task", "Unknown"),
                    "Trial": metadata.get("trial_name", "Trial"),
                    "Project": metadata.get("project_name", project_path.name),
                    "motion_raw.csv": "Yes" if file_exists(project_path, "motion_raw.csv") else "No",
                    "motion_filtered.csv": "Yes" if file_exists(project_path, "motion_filtered.csv") else "No",
                    "normalized_curves.csv": "Yes" if file_exists(project_path, "normalized_curves.csv") else "No",
                }
            )

        st.dataframe(
            pd.DataFrame(selected_table),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Primary Trial Workflow Status")
        show_workflow_status(selected_project_paths[0])

        can_continue = True

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------

st.divider()

workflow_navigation(
    back_step=None,
    next_step="preprocessing",
    next_enabled=can_continue,
    next_disabled_message="Process at least one video before continuing.",
)