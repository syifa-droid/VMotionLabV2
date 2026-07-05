import time
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from modules.app_style import apply_global_style
from modules.camera_manager import (
    create_video_writer,
    get_camera_properties,
    list_available_cameras,
    test_camera,
)
from modules.motion_engine import (
    DEFAULT_POSE_MODEL_PATH,
    POSE_MODEL_DOWNLOAD_URL,
    MotionEngine,
    get_angle_convention,
    pose_model_exists,
)
from modules.project_manager import (
    file_exists,
    load_metadata,
    save_metadata,
)
from modules.state import initialize_state, require_active_project
from modules.ui_navigation import (
    show_project_header,
    show_workflow_status,
    workflow_navigation,
)


st.set_page_config(
    page_title="Camera | VMotionLab",
    layout="wide",
)
apply_global_style()

initialize_state()
project_path = require_active_project()

st.title("Camera")
st.caption("Record video and capture markerless clinical motion data.")

show_project_header()

st.info(
    "VMotionLab currently assumes sagittal-plane movement analysis for walking. "
    "Position static camera in 30 or 45 degrees to optimally capture contralateral side. "
)

st.warning(
    "Angle accuracy may be affected by camera position, participant alignment, "
    "clothing, lighting, occlusion, landmark visibility, movement speed, and "
    "out-of-plane motion."
)

if not pose_model_exists():
    st.error(
        "MediaPipe Pose Landmarker model is missing. "
        "Run `python scripts/download_pose_model.py` from the VMotionLab folder."
    )
    st.code("python scripts/download_pose_model.py")
    st.caption(f"Expected model path: {DEFAULT_POSE_MODEL_PATH}")
    st.caption(f"Model source: {POSE_MODEL_DOWNLOAD_URL}")
    st.stop()

st.divider()


# -----------------------------------------------------------------------------
# Camera settings
# -----------------------------------------------------------------------------

st.markdown("### Camera Settings")

with st.expander("Find available cameras", expanded=False):
    st.write(
        "Use this if you are not sure which camera index belongs to your webcam, "
        "DroidCam, OBS Virtual Camera, or external camera."
    )

    max_index = st.number_input(
        "Maximum camera index to scan",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
    )

    if st.button("Scan Cameras"):
        with st.spinner("Scanning camera indices..."):
            cameras = list_available_cameras(max_index=int(max_index))

        if cameras:
            st.success(f"Available camera indices: {cameras}")
        else:
            st.error("No working camera was found.")

col1, col2, col3 = st.columns(3)

with col1:
    camera_index = st.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Try 0 for laptop webcam. OBS Virtual Camera or DroidCam may use another index.",
    )

with col2:
    recording_duration = st.number_input(
        "Recording Duration (seconds)",
        min_value=2,
        max_value=300,
        value=10,
        step=1,
    )

with col3:
    target_fps = st.number_input(
        "Video Save FPS",
        min_value=5,
        max_value=60,
        value=30,
        step=1,
    )

col4, col5, col6 = st.columns(3)

with col4:
    mirror_camera = st.checkbox(
        "Mirror camera preview",
        value=False,
        help="Use this if the camera view appears reversed.",
    )

with col5:
    draw_landmarks = st.checkbox(
        "Show landmarks on video",
        value=True,
        help="If enabled, the saved recording.mp4 will include landmark overlay.",
    )

with col6:
    include_debug_angles = st.checkbox(
        "Include debug geometric angles",
        value=False,
        help="For development only. Clinical users normally do not need this.",
    )


camera_properties = get_camera_properties(int(camera_index))

with st.container(border=True):
    st.write("**Camera properties**")
    st.write(
        f"Reported width: `{camera_properties['width']:.0f}` px  \n"
        f"Reported height: `{camera_properties['height']:.0f}` px  \n"
        f"Reported FPS: `{camera_properties['fps']:.2f}`"
    )


# -----------------------------------------------------------------------------
# Preview
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Camera Preview")

preview_placeholder = st.empty()

if st.button("Test Camera / Show One Frame"):
    if not test_camera(int(camera_index)):
        st.error(f"Camera index {camera_index} could not be opened.")
    else:
        engine = MotionEngine(
            camera_index=int(camera_index),
            include_debug_angles=include_debug_angles,
            draw_landmarks=draw_landmarks,
            mirror_camera=mirror_camera,
        )

        try:
            opened = engine.start_camera()

            if not opened:
                st.error("Could not open camera.")
            else:
                frame, data = engine.get_frame()

                if frame is None:
                    st.error("Could not read frame from camera.")
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(
                        frame_rgb,
                        channels="RGB",
                        caption="Camera preview",
                        use_container_width=True,
                    )

                    if data and data.get("pose_detected"):
                        st.success("Pose detected.")
                    else:
                        st.warning("No pose detected in this frame.")

        finally:
            engine.release()


# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Record Trial")

st.write(
    "Click **Record Trial** to record for the selected duration. "
    "After recording, VMotionLab will save `recording.mp4`, `landmarks.csv`, "
    "`motion_raw.csv`, and update `metadata.json`."
)

record_button = st.button(
    "Record Trial",
    type="primary",
    use_container_width=True,
)

if record_button:
    if not test_camera(int(camera_index)):
        st.error(f"Camera index {camera_index} could not be opened.")

    else:
        recording_path = project_path / "recording.mp4"
        motion_path = project_path / "motion_raw.csv"
        landmarks_path = project_path / "landmarks.csv"

        engine = MotionEngine(
            camera_index=int(camera_index),
            include_debug_angles=include_debug_angles,
            draw_landmarks=draw_landmarks,
            mirror_camera=mirror_camera,
        )

        video_writer = None
        progress_bar = st.progress(0)
        status_box = st.empty()
        live_view = st.empty()

        frame_count = 0
        detected_count = 0
        start_time = time.time()

        try:
            opened = engine.start_camera()

            if not opened:
                st.error("Could not open camera.")
                st.stop()

            engine.start_recording(reset_buffers=True)

            status_box.info("Recording started...")

            while True:
                elapsed = time.time() - start_time

                if elapsed >= float(recording_duration):
                    break

                if engine.cap is None:
                    st.error("Camera unexpectedly disconnected.")
                    break

                success, raw_frame = engine.cap.read()

                if not success or raw_frame is None:
                    st.error("Could not read frame from camera.")
                    break

                result = engine.process_frame(
                    frame_bgr=raw_frame,
                    frame_index=frame_count,
                    time_seconds=elapsed,
                )

                output_frame = result.annotated_frame

                if output_frame is None:
                    output_frame = raw_frame

                if not draw_landmarks:
                    if mirror_camera:
                        output_frame = cv2.flip(raw_frame, 1)
                    else:
                        output_frame = raw_frame

                if video_writer is None:
                    height, width = output_frame.shape[:2]
                    video_writer = create_video_writer(
                        output_path=recording_path,
                        fps=float(target_fps),
                        frame_width=width,
                        frame_height=height,
                    )

                video_writer.write(output_frame)

                frame_count += 1

                if result.detected:
                    detected_count += 1

                if frame_count % 3 == 0:
                    frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    live_view.image(
                        frame_rgb,
                        channels="RGB",
                        caption="Recording preview",
                        use_container_width=True,
                    )

                progress = min(elapsed / float(recording_duration), 1.0)
                progress_bar.progress(progress)

            engine.stop_recording()

            if video_writer is not None:
                video_writer.release()

            engine.save_motion_csv(motion_path)
            engine.save_landmarks_csv(landmarks_path)

            end_time = time.time()
            actual_duration = end_time - start_time
            actual_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
            detection_rate = (
                detected_count / frame_count * 100.0
                if frame_count > 0
                else 0.0
            )

            metadata = load_metadata(project_path)

            if metadata is not None:
                metadata["current_step"] = "camera"
                metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

                metadata["camera"] = {
                    "camera_index": int(camera_index),
                    "reported_width": camera_properties["width"],
                    "reported_height": camera_properties["height"],
                    "reported_fps": camera_properties["fps"],
                    "target_save_fps": float(target_fps),
                    "mirror_camera": bool(mirror_camera),
                    "draw_landmarks_on_video": bool(draw_landmarks),
                }

                metadata["recording"] = {
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                    "requested_duration_seconds": float(recording_duration),
                    "actual_duration_seconds": float(actual_duration),
                    "recorded_frames": int(frame_count),
                    "actual_processing_fps": float(actual_fps),
                    "pose_detected_frames": int(detected_count),
                    "pose_detection_rate_percent": float(detection_rate),
                }

                metadata["angle_convention"] = get_angle_convention()

                metadata["files"]["recording_video"] = "recording.mp4"
                metadata["files"]["landmarks"] = "landmarks.csv"
                metadata["files"]["motion_raw"] = "motion_raw.csv"

                save_metadata(project_path, metadata)

            progress_bar.progress(1.0)
            status_box.success("Recording completed and files saved.")

            st.success("Camera step completed.")

            summary_df = pd.DataFrame(
                [
                    {
                        "Output": "recording.mp4",
                        "Status": "Saved" if recording_path.exists() else "Missing",
                    },
                    {
                        "Output": "landmarks.csv",
                        "Status": "Saved" if landmarks_path.exists() else "Missing",
                    },
                    {
                        "Output": "motion_raw.csv",
                        "Status": "Saved" if motion_path.exists() else "Missing",
                    },
                    {
                        "Output": "metadata.json",
                        "Status": "Updated"
                        if (project_path / "metadata.json").exists()
                        else "Missing",
                    },
                ]
            )

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.write(
                f"**Recorded frames:** {frame_count}  \n"
                f"**Actual processing FPS:** {actual_fps:.2f}  \n"
                f"**Pose detection rate:** {detection_rate:.1f}%"
            )

        finally:
            if video_writer is not None:
                video_writer.release()

            engine.release()


# -----------------------------------------------------------------------------
# Output validation
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Camera Output Files")

expected_files = {
    "Recording video": "recording.mp4",
    "Landmark data": "landmarks.csv",
    "Raw clinical motion data": "motion_raw.csv",
    "Metadata": "metadata.json",
}

for label, filename in expected_files.items():
    if file_exists(project_path, filename):
        st.success(f"{label}: `{filename}` found")
    else:
        st.warning(f"{label}: `{filename}` not found yet")


st.divider()
st.markdown("### Workflow Status")
show_workflow_status(project_path)

camera_complete = (
    file_exists(project_path, "recording.mp4")
    and file_exists(project_path, "landmarks.csv")
    and file_exists(project_path, "motion_raw.csv")
)

workflow_navigation(
    back_step="new_analysis",
    next_step="trim_trial",
    next_enabled=camera_complete,
    next_disabled_message=(
        "Camera step is not complete yet. `recording.mp4`, `landmarks.csv`, "
        "and `motion_raw.csv` are required before continuing."
    ),
)