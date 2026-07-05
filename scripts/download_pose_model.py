"""
Download MediaPipe Pose Landmarker model for VMotionLab.

This script downloads the lightweight pose landmarker task model into:

    models/pose_landmarker_lite.task

The model file is not committed to GitHub by default.
"""

from pathlib import Path
from urllib.request import urlretrieve


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Pose Landmarker model...")
    print(f"URL: {MODEL_URL}")
    print(f"Output: {MODEL_PATH}")

    urlretrieve(MODEL_URL, MODEL_PATH)

    print("Download complete.")


if __name__ == "__main__":
    main()