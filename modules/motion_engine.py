"""
VMotionLab - Motion Engine
==========================

Sprint 1 revised engine using the modern MediaPipe Tasks Pose Landmarker API.

This version does NOT use:

    mp.solutions.pose

Instead, it uses:

    mediapipe.tasks.python.vision.PoseLandmarker

Core principles
---------------
1. Separate geometric angles from clinical angles.
2. Export clinician-facing sagittal clinical angles by default.
3. Knee and elbow flexion use:

       clinical_flexion = 180 - geometric_angle

4. Ankle is handled separately, with neutral ankle position approximately 90°.
5. Wrist and ankle are reported with caution because they are sensitive to
   landmark visibility and segment orientation.
6. The engine assumes sagittal-plane movement.
7. The reported values are clinical sagittal-plane approximations, not full 3D
   joint kinematics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

POSE_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


# -----------------------------------------------------------------------------
# MediaPipe Pose landmark indices
# -----------------------------------------------------------------------------

LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    # Face / head simplified
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),

    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),

    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


# -----------------------------------------------------------------------------
# Output columns
# -----------------------------------------------------------------------------

CLINICAL_ANGLE_COLUMNS = [
    "trunk_flexion",
    "hip_flexion_r",
    "knee_flexion_r",
    "ankle_angle_r",
    "shoulder_flexion_r",
    "elbow_flexion_r",
    "wrist_flexion_r",
    "hip_flexion_l",
    "knee_flexion_l",
    "ankle_angle_l",
    "shoulder_flexion_l",
    "elbow_flexion_l",
    "wrist_flexion_l",
]

GEOMETRIC_DEBUG_COLUMNS = [
    "hip_geometric_r",
    "knee_geometric_r",
    "ankle_geometric_r",
    "shoulder_geometric_r",
    "elbow_geometric_r",
    "wrist_geometric_r",
    "hip_geometric_l",
    "knee_geometric_l",
    "ankle_geometric_l",
    "shoulder_geometric_l",
    "elbow_geometric_l",
    "wrist_geometric_l",
]

ANGLE_CONVENTION = {
    "plane_assumption": "sagittal",
    "angle_type": "clinical sagittal-plane approximation",
    "neutral_values": {
        "trunk_flexion": 0,
        "hip_flexion": 0,
        "knee_flexion": 0,
        "shoulder_flexion": 0,
        "elbow_flexion": 0,
        "wrist_flexion": 0,
        "ankle_angle": 90,
    },
    "notes": [
        "Knee and elbow flexion are calculated as 180 degrees minus the raw geometric joint angle.",
        "Ankle is reported as the raw shank-foot angle, where neutral is approximately 90 degrees.",
        "Wrist is reported as sagittal deviation from forearm-hand alignment and should be interpreted cautiously.",
        "Angles are derived from 2D landmark positions and should not be interpreted as full 3D joint kinematics.",
    ],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pose_model_exists(model_path: Path | str = DEFAULT_POSE_MODEL_PATH) -> bool:
    """Return True if the MediaPipe pose model file exists."""
    return Path(model_path).exists()


def get_pose_model_message(model_path: Path | str = DEFAULT_POSE_MODEL_PATH) -> str:
    """Return a user-friendly model missing message."""
    return (
        f"MediaPipe Pose Landmarker model not found:\n\n"
        f"{Path(model_path)}\n\n"
        f"Run this command from the VMotionLab folder:\n\n"
        f"python scripts/download_pose_model.py"
    )


def get_angle_convention() -> Dict[str, Any]:
    """Return VMotionLab clinical angle convention metadata."""
    return ANGLE_CONVENTION.copy()


def create_empty_motion_dataframe(include_debug_angles: bool = False) -> pd.DataFrame:
    """Return an empty motion dataframe with the expected columns."""
    columns = ["frame", "time", "pose_detected"] + CLINICAL_ANGLE_COLUMNS

    if include_debug_angles:
        columns += GEOMETRIC_DEBUG_COLUMNS

    return pd.DataFrame(columns=columns)


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass
class MotionFrameResult:
    """Container for one processed frame."""

    frame_index: int
    time_seconds: float
    detected: bool
    motion_row: Dict[str, Any]
    landmark_rows: List[Dict[str, Any]] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Clinical angle calculator
# -----------------------------------------------------------------------------

class ClinicalAngleCalculator:
    """
    Calculate clinical sagittal angles from MediaPipe pose landmarks.

    The calculator uses normalized 2D image landmarks.

    For extension-type joints:

        clinical_flexion = 180 - geometric_angle

    Examples:
        geometric knee angle 180° -> knee flexion 0°
        geometric knee angle 120° -> knee flexion 60°
    """

    def __init__(self, min_visibility: float = 0.50) -> None:
        self.min_visibility = float(min_visibility)

    @staticmethod
    def _safe_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return float("nan")

        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)

        return float(np.degrees(np.arccos(cosine)))

    @classmethod
    def angle_at_joint(
        cls,
        proximal: np.ndarray,
        joint: np.ndarray,
        distal: np.ndarray,
    ) -> float:
        vector_1 = proximal - joint
        vector_2 = distal - joint
        return cls._safe_angle_between_vectors(vector_1, vector_2)

    @staticmethod
    def _clinical_from_extension_angle(geometric_angle: float) -> float:
        if np.isnan(geometric_angle):
            return float("nan")

        return float(180.0 - geometric_angle)

    def _point(
        self,
        landmarks: List[Any],
        landmark_index: int,
    ) -> Optional[np.ndarray]:
        try:
            lm = landmarks[landmark_index]
        except (IndexError, TypeError):
            return None

        visibility = getattr(lm, "visibility", 1.0)

        if visibility is not None and visibility < self.min_visibility:
            return None

        return np.array([float(lm.x), float(lm.y)], dtype=float)

    def _midpoint(
        self,
        landmarks: List[Any],
        first_index: int,
        second_index: int,
    ) -> Optional[np.ndarray]:
        p1 = self._point(landmarks, first_index)
        p2 = self._point(landmarks, second_index)

        if p1 is None or p2 is None:
            return None

        return (p1 + p2) / 2.0

    def _calculate_side(
        self,
        landmarks: List[Any],
        side: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        if side == "r":
            shoulder = RIGHT_SHOULDER
            elbow = RIGHT_ELBOW
            wrist = RIGHT_WRIST
            index = RIGHT_INDEX
            hip = RIGHT_HIP
            knee = RIGHT_KNEE
            ankle = RIGHT_ANKLE
            foot_index = RIGHT_FOOT_INDEX
        elif side == "l":
            shoulder = LEFT_SHOULDER
            elbow = LEFT_ELBOW
            wrist = LEFT_WRIST
            index = LEFT_INDEX
            hip = LEFT_HIP
            knee = LEFT_KNEE
            ankle = LEFT_ANKLE
            foot_index = LEFT_FOOT_INDEX
        else:
            raise ValueError("side must be 'r' or 'l'")

        p_shoulder = self._point(landmarks, shoulder)
        p_elbow = self._point(landmarks, elbow)
        p_wrist = self._point(landmarks, wrist)
        p_index = self._point(landmarks, index)
        p_hip = self._point(landmarks, hip)
        p_knee = self._point(landmarks, knee)
        p_ankle = self._point(landmarks, ankle)
        p_foot_index = self._point(landmarks, foot_index)

        geometric: Dict[str, float] = {
            f"hip_geometric_{side}": float("nan"),
            f"knee_geometric_{side}": float("nan"),
            f"ankle_geometric_{side}": float("nan"),
            f"shoulder_geometric_{side}": float("nan"),
            f"elbow_geometric_{side}": float("nan"),
            f"wrist_geometric_{side}": float("nan"),
        }

        clinical: Dict[str, float] = {
            f"hip_flexion_{side}": float("nan"),
            f"knee_flexion_{side}": float("nan"),
            f"ankle_angle_{side}": float("nan"),
            f"shoulder_flexion_{side}": float("nan"),
            f"elbow_flexion_{side}": float("nan"),
            f"wrist_flexion_{side}": float("nan"),
        }

        # Hip: shoulder -> hip -> knee
        if p_shoulder is not None and p_hip is not None and p_knee is not None:
            hip_geom = self.angle_at_joint(p_shoulder, p_hip, p_knee)
            geometric[f"hip_geometric_{side}"] = hip_geom
            clinical[f"hip_flexion_{side}"] = self._clinical_from_extension_angle(
                hip_geom
            )

        # Knee: hip -> knee -> ankle
        if p_hip is not None and p_knee is not None and p_ankle is not None:
            knee_geom = self.angle_at_joint(p_hip, p_knee, p_ankle)
            geometric[f"knee_geometric_{side}"] = knee_geom
            clinical[f"knee_flexion_{side}"] = self._clinical_from_extension_angle(
                knee_geom
            )

        # Ankle: knee -> ankle -> foot index
        # Neutral is approximately 90°.
        if p_knee is not None and p_ankle is not None and p_foot_index is not None:
            ankle_geom = self.angle_at_joint(p_knee, p_ankle, p_foot_index)
            geometric[f"ankle_geometric_{side}"] = ankle_geom
            clinical[f"ankle_angle_{side}"] = ankle_geom

        # Shoulder: hip -> shoulder -> elbow
        # Arm at side is approximately 0°.
        if p_hip is not None and p_shoulder is not None and p_elbow is not None:
            shoulder_geom = self.angle_at_joint(p_hip, p_shoulder, p_elbow)
            geometric[f"shoulder_geometric_{side}"] = shoulder_geom
            clinical[f"shoulder_flexion_{side}"] = shoulder_geom

        # Elbow: shoulder -> elbow -> wrist
        if p_shoulder is not None and p_elbow is not None and p_wrist is not None:
            elbow_geom = self.angle_at_joint(p_shoulder, p_elbow, p_wrist)
            geometric[f"elbow_geometric_{side}"] = elbow_geom
            clinical[f"elbow_flexion_{side}"] = self._clinical_from_extension_angle(
                elbow_geom
            )

        # Wrist: elbow -> wrist -> index
        if p_elbow is not None and p_wrist is not None and p_index is not None:
            wrist_geom = self.angle_at_joint(p_elbow, p_wrist, p_index)
            geometric[f"wrist_geometric_{side}"] = wrist_geom
            clinical[f"wrist_flexion_{side}"] = self._clinical_from_extension_angle(
                wrist_geom
            )

        return clinical, geometric

    def _calculate_trunk(self, landmarks: List[Any]) -> float:
        shoulder_mid = self._midpoint(
            landmarks,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
        )
        hip_mid = self._midpoint(
            landmarks,
            LEFT_HIP,
            RIGHT_HIP,
        )

        if shoulder_mid is None or hip_mid is None:
            return float("nan")

        trunk_vector = shoulder_mid - hip_mid

        # In image coordinates, y increases downward, so vertical up is [0, -1].
        vertical_up = np.array([0.0, -1.0], dtype=float)

        return self._safe_angle_between_vectors(trunk_vector, vertical_up)

    def calculate(self, landmarks: List[Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        clinical: Dict[str, float] = {
            column: float("nan") for column in CLINICAL_ANGLE_COLUMNS
        }

        geometric: Dict[str, float] = {
            column: float("nan") for column in GEOMETRIC_DEBUG_COLUMNS
        }

        clinical["trunk_flexion"] = self._calculate_trunk(landmarks)

        right_clinical, right_geometric = self._calculate_side(landmarks, "r")
        left_clinical, left_geometric = self._calculate_side(landmarks, "l")

        clinical.update(right_clinical)
        clinical.update(left_clinical)
        geometric.update(right_geometric)
        geometric.update(left_geometric)

        return clinical, geometric


# -----------------------------------------------------------------------------
# Landmark recorder
# -----------------------------------------------------------------------------

class LandmarkRecorder:
    """Convert MediaPipe Tasks landmarks into CSV-ready rows."""

    @staticmethod
    def _side_from_name(name: str) -> str:
        if name.startswith("left_"):
            return "left"
        if name.startswith("right_"):
            return "right"
        return "midline"

    def rows_from_landmarks(
        self,
        landmarks: Optional[List[Any]],
        frame_index: int,
        time_seconds: float,
    ) -> List[Dict[str, Any]]:
        if landmarks is None:
            return []

        rows: List[Dict[str, Any]] = []

        for index, lm in enumerate(landmarks):
            name = LANDMARK_NAMES.get(index, f"landmark_{index}")

            rows.append(
                {
                    "frame": int(frame_index),
                    "time": float(time_seconds),
                    "landmark_index": int(index),
                    "landmark_name": name,
                    "side": self._side_from_name(name),
                    "x": float(getattr(lm, "x", np.nan)),
                    "y": float(getattr(lm, "y", np.nan)),
                    "z": float(getattr(lm, "z", np.nan)),
                    "visibility": float(getattr(lm, "visibility", np.nan)),
                    "presence": float(getattr(lm, "presence", np.nan)),
                }
            )

        return rows


# -----------------------------------------------------------------------------
# Drawing helper
# -----------------------------------------------------------------------------

def draw_pose_landmarks(
    frame_bgr: np.ndarray,
    landmarks: List[Any],
    min_visibility: float = 0.50,
) -> np.ndarray:
    """
    Draw pose landmarks manually using OpenCV.

    This avoids dependency on the old mp.solutions.drawing_utils API.
    """
    output = frame_bgr.copy()
    height, width = output.shape[:2]

    points: Dict[int, Tuple[int, int]] = {}

    for index, lm in enumerate(landmarks):
        visibility = getattr(lm, "visibility", 1.0)

        if visibility is not None and visibility < min_visibility:
            continue

        x = int(float(lm.x) * width)
        y = int(float(lm.y) * height)

        points[index] = (x, y)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(output, points[start], points[end], (0, 255, 0), 2)

    for _, point in points.items():
        cv2.circle(output, point, 4, (0, 0, 255), -1)

    return output


# -----------------------------------------------------------------------------
# Motion Engine
# -----------------------------------------------------------------------------

class MotionEngine:
    """
    Core VMotionLab motion engine using MediaPipe Tasks Pose Landmarker.

    Typical Camera page use:

        engine = MotionEngine(camera_index=0)
        engine.start_camera()
        engine.start_recording()

        frame, data = engine.get_frame()

        engine.stop_recording()
        engine.save_motion_csv(project_path / "motion_raw.csv")
        engine.save_landmarks_csv(project_path / "landmarks.csv")
        engine.release()
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Path | str = DEFAULT_POSE_MODEL_PATH,
        min_detection_confidence: float = 0.50,
        min_pose_presence_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        min_visibility: float = 0.50,
        include_debug_angles: bool = False,
        draw_landmarks: bool = True,
        mirror_camera: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
    ) -> None:
        self.camera_index = int(camera_index)
        self.model_path = Path(model_path)

        self.min_detection_confidence = float(min_detection_confidence)
        self.min_pose_presence_confidence = float(min_pose_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.min_visibility = float(min_visibility)

        self.include_debug_angles = bool(include_debug_angles)
        self.draw_landmarks = bool(draw_landmarks)
        self.mirror_camera = bool(mirror_camera)

        # Kept for backward compatibility with earlier constructor calls.
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks

        if not self.model_path.exists():
            raise FileNotFoundError(get_pose_model_message(self.model_path))

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        self.angle_calculator = ClinicalAngleCalculator(
            min_visibility=self.min_visibility,
        )
        self.landmark_recorder = LandmarkRecorder()

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        self.start_time = time.time()
        self._last_timestamp_ms = -1

        self.is_recording = False
        self.motion_rows: List[Dict[str, Any]] = []
        self.landmark_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Camera handling
    # ------------------------------------------------------------------

    def start_camera(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_index)
        return bool(self.cap.isOpened())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.landmarker is not None:
            self.landmarker.close()

    def reset_timing(self) -> None:
        self.frame_index = 0
        self.start_time = time.time()
        self._last_timestamp_ms = -1

    def get_camera_fps(self) -> float:
        if self.cap is None:
            return float("nan")

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            return float("nan")

        return float(fps)

    # ------------------------------------------------------------------
    # Recording buffers
    # ------------------------------------------------------------------

    def start_recording(self, reset_buffers: bool = True) -> None:
        if reset_buffers:
            self.motion_rows = []
            self.landmark_rows = []
            self.reset_timing()

        self.is_recording = True

    def stop_recording(self) -> None:
        self.is_recording = False

    def clear_buffers(self) -> None:
        self.motion_rows = []
        self.landmark_rows = []

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _safe_timestamp_ms(self, time_seconds: float) -> int:
        timestamp_ms = int(float(time_seconds) * 1000.0)

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: Optional[int] = None,
        time_seconds: Optional[float] = None,
    ) -> MotionFrameResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr cannot be None")

        if self.mirror_camera:
            frame_bgr = cv2.flip(frame_bgr, 1)

        if frame_index is None:
            frame_index = self.frame_index

        if time_seconds is None:
            time_seconds = time.time() - self.start_time

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = self._safe_timestamp_ms(float(time_seconds))
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        detected = bool(result.pose_landmarks and len(result.pose_landmarks) > 0)

        landmarks = result.pose_landmarks[0] if detected else None

        motion_row: Dict[str, Any] = {
            "frame": int(frame_index),
            "time": float(time_seconds),
            "pose_detected": bool(detected),
        }

        landmark_rows: List[Dict[str, Any]] = []

        if detected and landmarks is not None:
            clinical_angles, geometric_angles = self.angle_calculator.calculate(
                landmarks
            )
            motion_row.update(clinical_angles)

            if self.include_debug_angles:
                motion_row.update(geometric_angles)

            landmark_rows = self.landmark_recorder.rows_from_landmarks(
                landmarks=landmarks,
                frame_index=int(frame_index),
                time_seconds=float(time_seconds),
            )
        else:
            for column in CLINICAL_ANGLE_COLUMNS:
                motion_row[column] = float("nan")

            if self.include_debug_angles:
                for column in GEOMETRIC_DEBUG_COLUMNS:
                    motion_row[column] = float("nan")

        if self.draw_landmarks and detected and landmarks is not None:
            annotated_frame = draw_pose_landmarks(
                frame_bgr,
                landmarks,
                min_visibility=self.min_visibility,
            )
        else:
            annotated_frame = frame_bgr.copy()

        frame_result = MotionFrameResult(
            frame_index=int(frame_index),
            time_seconds=float(time_seconds),
            detected=bool(detected),
            motion_row=motion_row,
            landmark_rows=landmark_rows,
            annotated_frame=annotated_frame,
        )

        if self.is_recording:
            self.motion_rows.append(motion_row)
            self.landmark_rows.extend(landmark_rows)

        self.frame_index = int(frame_index) + 1

        return frame_result

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if self.cap is None or not self.cap.isOpened():
            opened = self.start_camera()
            if not opened:
                return None, None

        assert self.cap is not None

        success, frame = self.cap.read()

        if not success or frame is None:
            return None, None

        result = self.process_frame(frame)
        return result.annotated_frame, result.motion_row

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def get_motion_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.motion_rows)

    def get_landmarks_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.landmark_rows)

    def save_motion_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_motion_dataframe()

        base_columns = ["frame", "time", "pose_detected"]
        desired_columns = base_columns + CLINICAL_ANGLE_COLUMNS

        if self.include_debug_angles:
            desired_columns += GEOMETRIC_DEBUG_COLUMNS

        for column in desired_columns:
            if column not in df.columns:
                df[column] = np.nan

        df = df[desired_columns]
        df.to_csv(path, index=False)
        return path

    def save_landmarks_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_landmarks_dataframe()

        desired_columns = [
            "frame",
            "time",
            "landmark_index",
            "landmark_name",
            "side",
            "x",
            "y",
            "z",
            "visibility",
            "presence",
        ]

        for column in desired_columns:
            if column not in df.columns:
                df[column] = np.nan

        df = df[desired_columns]
        df.to_csv(path, index=False)
        return path

    def save_outputs(self, project_path: Path | str) -> Dict[str, Path]:
        project_path = Path(project_path)
        project_path.mkdir(parents=True, exist_ok=True)

        motion_path = self.save_motion_csv(project_path / "motion_raw.csv")
        landmarks_path = self.save_landmarks_csv(project_path / "landmarks.csv")

        return {
            "motion_raw": motion_path,
            "landmarks": landmarks_path,
        }

    # ------------------------------------------------------------------
    # Offline video processing
    # ------------------------------------------------------------------

    def process_video_file(
        self,
        video_path: Path | str,
        save_project_path: Optional[Path | str] = None,
        max_frames: Optional[int] = None,
    ) -> pd.DataFrame:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            fps = 30.0

        self.clear_buffers()
        self.is_recording = True
        self._last_timestamp_ms = -1

        frame_index = 0

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                success, frame = cap.read()

                if not success or frame is None:
                    break

                time_seconds = frame_index / float(fps)

                self.process_frame(
                    frame_bgr=frame,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                )

                frame_index += 1

        finally:
            cap.release()
            self.is_recording = False

        if save_project_path is not None:
            self.save_outputs(save_project_path)

        return self.get_motion_dataframe()