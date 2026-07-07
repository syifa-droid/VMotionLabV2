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

## License

This project is licensed under the MIT License

## Disclaimer

VMotionLabV2 is intended for research, education, and workflow visualization.
Clinical interpretation and validation remain the responsibility of the user, researcher, or qualified clinician.

## Acknowledgements

VMotionLabV2 is developed with the support of several open-source libraries and tools. We gratefully acknowledge the developers, maintainers, and contributor communities behind these projects.

This project uses **MediaPipe** by Google for markerless human pose estimation and landmark detection. MediaPipe provides the core pose-tracking capability that enables VMotionLabV2 to detect body landmarks from video and camera input.

VMotionLabV2 also makes use of the following open-source Python libraries:

* **OpenCV** for video capture, frame processing, and computer vision utilities.
* **Streamlit** for building the interactive web-based user interface.
* **NumPy** for numerical computation.
* **pandas** for tabular data handling and CSV export.
* **SciPy** for signal processing and filtering.
* **Matplotlib** and/or **Plotly** for visualization of kinematic data and analysis results.

We sincerely thank the open-source community for making these tools publicly available. VMotionLabV2 is an independent project and is not officially affiliated with, endorsed by, or sponsored by Google, MediaPipe, OpenCV, Streamlit, NumPy, pandas, SciPy, Matplotlib, or Plotly.

All third-party libraries remain the property of their respective authors and are distributed under their respective open-source licenses. Users and contributors should refer to each library’s official license for detailed terms of use.


## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install streamlit
pip install -r requirements.txt

