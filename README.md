# VMotionLabV2

VMotionLabV2 is a Streamlit-based markerless video motion analysis app.

The app supports uploaded trimmed videos, offline pose processing, joint-angle preprocessing, kinematic analysis, multi-trial comparison, and PDF report generation.

## Workflow

1. Upload one or more trimmed videos
2. Assign participant, movement/task, side, and trial name
3. Process pose data
4. Preprocess selected joint angles
5. Run kinematics or multi-trial comparison
6. Generate report

## Notes

- VMotionLabV2 uses uploaded videos only.
- Live camera recording has been removed in this simplified version.
- Each uploaded video becomes one trial.
- Multi-trial comparison is integrated into the Kinematics page.
- This tool is intended for research, education, and workflow visualization.
- Clinical validation and interpretation remain the responsibility of the data taker or clinician.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## License

This project is licensed under the MIT License.

## Disclaimer

VMotionLabV2 is intended for research, education, and workflow visualization.
It is not a diagnostic medical device. Clinical interpretation and validation
remain the responsibility of the user, researcher, or qualified clinician.