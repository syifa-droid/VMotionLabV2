"""
VMotionLab - Clinical Notes
===========================

Centralized clinical notes, angle conventions, and limitations.

Keeping these statements in one module ensures that the Kinematics page,
Report page, and generated PDF use consistent professional language.
"""

from __future__ import annotations

import streamlit as st


ANGLE_CONVENTION_TEXT = """
### Clinical Angle Convention

- VMotionLab currently assumes sagittal-plane movement analysis for walking.
- Hip, knee, shoulder, elbow, wrist, and trunk angles are reported as clinical sagittal-plane approximations in walking.
- For hip, knee, shoulder, elbow, wrist, and trunk, neutral posture is treated as approximately 0°.
- Knee and elbow flexion are calculated from the geometric joint angle so that full extension is approximately 0°.
- The ankle is handled differently: neutral ankle position is approximately 90°.
- Wrist and ankle values should be interpreted carefully because they are sensitive to landmark visibility, hand/foot orientation, and sagittal alignment.
"""


ANGLE_LIMITATION_TEXT = """
### Angle Calculation Limitations

VMotionLab currently estimates joint angles using markerless 2D landmark detection and assumes that the recorded movement is primarily performed in the sagittal plane. The reported angles are clinical sagittal-plane approximations derived from visible body landmarks and should not be interpreted as full three-dimensional joint kinematics.

Angle accuracy may be affected by camera position, participant alignment, clothing, landmark visibility, occlusion, movement speed, lighting condition, and out-of-plane motion. Wrist and ankle angles should be interpreted with particular caution because these joints are more sensitive to landmark detection variability and segment orientation.

VMotionLab is intended to support clinical observation, education, documentation, and research screening. The results should be interpreted by qualified clinicians and should not replace comprehensive clinical assessment or laboratory-based motion analysis when high-precision measurement is required.
"""


REPORT_ANGLE_CONVENTION_TEXT = """
VMotionLab angle convention: sagittal-plane clinical approximation in walking. Hip, knee, shoulder, elbow, wrist, and trunk angles are interpreted with neutral posture as approximately 0°. The ankle is treated separately, with neutral ankle position approximately 90°. Knee and elbow flexion are calculated so that full extension is approximately 0°.
"""


REPORT_ANGLE_LIMITATION_TEXT = """
VMotionLab estimates joint angles using markerless 2D landmark detection and assumes that the recorded movement is primarily performed in the sagittal plane. The reported values are clinical sagittal-plane approximations and should not be interpreted as full three-dimensional joint kinematics.

Angle accuracy may be affected by camera position, participant alignment, clothing, landmark visibility, occlusion, movement speed, lighting condition, and out-of-plane motion. Wrist and ankle angles should be interpreted with particular caution because these joints are more sensitive to landmark detection variability and segment orientation.

VMotionLab is intended to support clinical observation, education, documentation, and research screening. Results should be interpreted by qualified clinicians and should not replace comprehensive clinical assessment or laboratory-based motion analysis when high-precision measurement is required.
"""


def show_angle_convention_note() -> None:
    """Display clinical angle convention on Streamlit pages."""
    st.info(ANGLE_CONVENTION_TEXT)


def show_angle_limitation_note() -> None:
    """Display professional angle calculation limitation on Streamlit pages."""
    st.warning(ANGLE_LIMITATION_TEXT)