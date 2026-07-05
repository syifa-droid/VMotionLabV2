"""
VMotionLabV2 - Video Processor
==============================

Offline video processing for uploaded videos.

Responsibilities:
- Read uploaded video file
- Run MediaPipe pose analysis frame by frame using MotionEngine
- Save motion_raw.csv
- Save landmarks.csv
- Optionally save annotated video with landmarks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from modules.motion_engine import MotionEngine


def _delete_old_outputs(output_project_path: Path) -> None:
    """
    Delete old processing outputs before reprocessing.
    This prevents old empty files from being treated as valid.
    """
    old_files = [
        "motion_raw.csv",
        "landmarks.csv",
        "recording_with_pose.mp4",
    ]

    for filename in old_files:
        path = output_project_path / filename

        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


def _enable_engine_recording(engine: MotionEngine) -> None:
    """
    Force MotionEngine into recording mode if it uses recording flags.

    Different MotionEngine versions may use different attribute names.
    This function safely enables common recording flags.
    """
    possible_flags = [
        "is_recording",
        "recording",
        "recording_active",
        "save_data",
        "collect_data",
        "store_data",
    ]

    for flag in possible_flags:
        if hasattr(engine, flag):
            try:
                setattr(engine, flag, True)
            except Exception:
                pass

    possible_methods = [
        "start_recording",
        "start_capture",
        "start",
        "reset_buffers",
        "clear_buffers",
    ]

    for method_name in possible_methods:
        if hasattr(engine, method_name):
            method = getattr(engine, method_name)

            try:
                method()
            except TypeError:
                pass
            except Exception:
                pass


def _disable_engine_recording(engine: MotionEngine) -> None:
    """
    Disable recording flags after processing.
    """
    possible_flags = [
        "is_recording",
        "recording",
        "recording_active",
        "save_data",
        "collect_data",
        "store_data",
    ]

    for flag in possible_flags:
        if hasattr(engine, flag):
            try:
                setattr(engine, flag, False)
            except Exception:
                pass

    possible_methods = [
        "stop_recording",
        "stop_capture",
        "stop",
    ]

    for method_name in possible_methods:
        if hasattr(engine, method_name):
            method = getattr(engine, method_name)

            try:
                method()
            except TypeError:
                pass
            except Exception:
                pass


def _safe_process_frame(
    engine: MotionEngine,
    frame: np.ndarray,
    timestamp_ms: int,
):
    """
    Call MotionEngine.process_frame with flexible signature support.
    """
    try:
        return engine.process_frame(frame, timestamp_ms=timestamp_ms)
    except TypeError:
        try:
            return engine.process_frame(frame, timestamp_ms)
        except TypeError:
            return engine.process_frame(frame)


def _extract_output_frame(process_result, original_frame: np.ndarray) -> np.ndarray:
    """
    Safely extract an image frame from MotionEngine.process_frame() output.
    """
    candidate = None

    if isinstance(process_result, np.ndarray):
        candidate = process_result

    elif isinstance(process_result, tuple):
        for item in process_result:
            if isinstance(item, np.ndarray):
                candidate = item
                break

    elif isinstance(process_result, dict):
        for key in ["frame", "annotated_frame", "output_frame", "image"]:
            value = process_result.get(key)

            if isinstance(value, np.ndarray):
                candidate = value
                break

    if candidate is None:
        candidate = original_frame

    if not isinstance(candidate, np.ndarray):
        candidate = original_frame

    if candidate.ndim == 2:
        candidate = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)

    if candidate.ndim == 3 and candidate.shape[2] == 4:
        candidate = cv2.cvtColor(candidate, cv2.COLOR_BGRA2BGR)

    if candidate.dtype != np.uint8:
        candidate = np.clip(candidate, 0, 255).astype(np.uint8)

    return candidate


def _extract_motion_data(process_result) -> Optional[Dict[str, Any]]:
    """
    Try to extract motion data dictionary from MotionEngine.process_frame() result.

    Expected possible outputs:
    - (frame, data)
    - (frame, data, extra)
    - {"data": data}
    - {"motion": data}
    - data dictionary directly
    """
    candidate = None

    if isinstance(process_result, tuple):
        for item in process_result:
            if isinstance(item, dict):
                candidate = item
                break

    elif isinstance(process_result, dict):
        if "data" in process_result and isinstance(process_result["data"], dict):
            candidate = process_result["data"]
        elif "motion" in process_result and isinstance(process_result["motion"], dict):
            candidate = process_result["motion"]
        else:
            candidate = process_result

    if not isinstance(candidate, dict):
        return None

    # Keep only simple scalar values
    clean_data = {}

    for key, value in candidate.items():
        if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
            clean_data[key] = value

    if not clean_data:
        return None

    return clean_data


def _resize_if_needed(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Ensure frame size matches VideoWriter size.
    """
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))

    return frame


def clean_motion_raw_csv_time(
    motion_csv_path: Path | str,
    fps: float,
) -> None:
    """
    Make sure motion_raw.csv has numeric frame and time columns.

    Trim Trial requires numeric time values.
    """
    motion_csv_path = Path(motion_csv_path)

    if not motion_csv_path.exists():
        raise FileNotFoundError(f"motion_raw.csv not found: {motion_csv_path}")

    df = pd.read_csv(motion_csv_path)

    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise RuntimeError("motion_raw.csv is empty.")

    if "frame" not in df.columns:
        df.insert(0, "frame", range(len(df)))
    else:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

        if df["frame"].isna().all():
            df["frame"] = range(len(df))
        else:
            df["frame"] = df["frame"].interpolate().bfill().ffill().astype(int)

    if "time" in df.columns:
        numeric_time = pd.to_numeric(df["time"], errors="coerce")

        if numeric_time.isna().all():
            try:
                numeric_time = pd.to_timedelta(
                    df["time"],
                    errors="coerce",
                ).dt.total_seconds()
            except Exception:
                numeric_time = pd.Series([np.nan] * len(df))

        if numeric_time.isna().all():
            numeric_time = pd.Series(np.arange(len(df)) / float(fps))
        else:
            numeric_time = numeric_time.interpolate().bfill().ffill()

        numeric_time = numeric_time - numeric_time.iloc[0]
        df["time"] = numeric_time.astype(float)

    else:
        df.insert(1, "time", np.arange(len(df)) / float(fps))

    df.to_csv(motion_csv_path, index=False)


def _write_fallback_motion_csv(
    motion_rows: list[Dict[str, Any]],
    motion_csv_path: Path,
    fps: float,
) -> None:
    """
    Write fallback motion_raw.csv if MotionEngine did not save its own data.
    """
    if not motion_rows:
        return

    df = pd.DataFrame(motion_rows)

    if "frame" not in df.columns:
        df.insert(0, "frame", range(len(df)))

    if "time" not in df.columns:
        df.insert(1, "time", np.arange(len(df)) / float(fps))

    df.to_csv(motion_csv_path, index=False)


def _file_has_rows(csv_path: Path) -> bool:
    """
    Check whether CSV exists and has at least one data row.
    """
    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
        return not df.empty
    except Exception:
        return False


def process_uploaded_video(
    video_path: Path | str,
    output_project_path: Path | str,
    save_annotated_video: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Process uploaded video and save VMotionLab motion outputs.
    """
    video_path = Path(video_path)
    output_project_path = Path(output_project_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_project_path.mkdir(parents=True, exist_ok=True)
    _delete_old_outputs(output_project_path)

    motion_csv_path = output_project_path / "motion_raw.csv"
    landmarks_csv_path = output_project_path / "landmarks.csv"
    annotated_video_path = output_project_path / "recording_with_pose.mp4"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height.")

    writer = None

    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_video_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            writer = None
            save_annotated_video = False

    engine = MotionEngine()
    _enable_engine_recording(engine)

    processed_frames = 0
    fallback_motion_rows = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp_ms = int((processed_frames / fps) * 1000)

            result = _safe_process_frame(
                engine=engine,
                frame=frame,
                timestamp_ms=timestamp_ms,
            )

            data = _extract_motion_data(result)

            if data is not None:
                data["frame"] = processed_frames
                data["time"] = processed_frames / float(fps)
                fallback_motion_rows.append(data)

            output_frame = _extract_output_frame(result, frame)
            output_frame = _resize_if_needed(output_frame, width, height)

            if writer is not None:
                writer.write(output_frame)

            processed_frames += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(processed_frames / total_frames, 1.0))

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        _disable_engine_recording(engine)

    if processed_frames == 0:
        raise RuntimeError("No video frames were processed.")

    # Save motion data from MotionEngine
    if hasattr(engine, "save_motion_csv"):
        try:
            engine.save_motion_csv(motion_csv_path)
        except Exception:
            pass

    # If MotionEngine did not save rows, use fallback rows from process_frame()
    if not _file_has_rows(motion_csv_path):
        _write_fallback_motion_csv(
            motion_rows=fallback_motion_rows,
            motion_csv_path=motion_csv_path,
            fps=fps,
        )

    if not _file_has_rows(motion_csv_path):
        raise RuntimeError(
            "motion_raw.csv is empty. No pose/motion rows were recorded. "
            "Check that the person is clearly visible in the video and that "
            "MotionEngine.process_frame() returns motion data."
        )

    clean_motion_raw_csv_time(motion_csv_path, fps=fps)

    # Save landmarks
    if hasattr(engine, "save_landmarks_csv"):
        try:
            engine.save_landmarks_csv(landmarks_csv_path)
        except Exception:
            pass

    if hasattr(engine, "close"):
        try:
            engine.close()
        except Exception:
            pass

    duration_seconds = processed_frames / fps if fps > 0 else 0

    summary = {
        "video_path": str(video_path),
        "project_path": str(output_project_path),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "processed_frames": int(processed_frames),
        "duration_seconds": float(duration_seconds),
        "width": int(width),
        "height": int(height),
        "motion_csv": str(motion_csv_path),
        "landmarks_csv": str(landmarks_csv_path),
        "annotated_video": str(annotated_video_path) if save_annotated_video else None,
    }

    return summary