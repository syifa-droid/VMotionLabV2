"""
VMotionLab - Camera Manager
===========================

Utility functions for camera selection and video recording support.

This module does not contain Streamlit UI code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2


def test_camera(camera_index: int) -> bool:
    """
    Check whether a camera index can be opened.

    Args:
        camera_index:
            OpenCV camera index.

    Returns:
        True if the camera opens and returns at least one frame.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        return False

    success, frame = cap.read()
    cap.release()

    return bool(success and frame is not None)


def list_available_cameras(max_index: int = 5) -> List[int]:
    """
    Scan available camera indices.

    Args:
        max_index:
            Highest camera index to test.

    Returns:
        List of working camera indices.
    """
    available = []

    for index in range(max_index + 1):
        if test_camera(index):
            available.append(index)

    return available


def get_camera_properties(camera_index: int) -> Dict[str, float]:
    """
    Return basic camera properties.

    Args:
        camera_index:
            OpenCV camera index.

    Returns:
        Dictionary with width, height, fps.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        return {
            "width": 0.0,
            "height": 0.0,
            "fps": 0.0,
        }

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return {
        "width": float(width),
        "height": float(height),
        "fps": float(fps),
    }


def create_video_writer(
    output_path: Path | str,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    """
    Create an MP4 video writer.

    Args:
        output_path:
            Output recording path.
        fps:
            Frames per second.
        frame_width:
            Video width.
        frame_height:
            Video height.

    Returns:
        OpenCV VideoWriter.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(fps),
        (int(frame_width), int(frame_height)),
    )