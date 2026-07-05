from datetime import datetime

import pandas as pd
import streamlit as st

from modules.app_style import apply_global_style
from modules.clinical_notes import (
    show_angle_convention_note,
    show_angle_limitation_note,
)
from modules.project_manager import (
    file_exists,
    load_metadata,
    open_project_folder,
    save_metadata,
)
from modules.report_generator import (
    REPORT_FILENAME,
    REPORT_INFO_FILENAME,
    generate_report,
)
from modules.state import initialize_state, require_active_project
from modules.ui_navigation import (
    show_project_header,
    show_workflow_status,
    workflow_navigation,
)


st.set_page_config(
    page_title="Report | VMotionLab",
    layout="wide",
)
apply_global_style()

initialize_state()
project_path = require_active_project()

st.title("Report")
st.caption("Generate a PDF report with metadata, processing information, statistics, and figures.")

show_project_header()

show_angle_convention_note()
show_angle_limitation_note()

st.divider()


# -----------------------------------------------------------------------------
# Report readiness
# -----------------------------------------------------------------------------

st.markdown("### Report Readiness")

required_files = {
    "Metadata": "metadata.json",
    "Statistics": "statistics.csv",
    "Normalized curves": "normalized_curves.csv",
    "Kinematics time figure": "figures/kinematics_time_curves.png",
    "Normalized curve figure": "figures/kinematics_normalized_curves.png",
}

optional_files = {
    "Landmarks": "landmarks.csv",
    "Filtered motion": "motion_filtered.csv",
    "Processing info": "processing_info.json",
    "Kinematics info": "kinematics_info.json",
}

required_status = []
optional_status = []

for label, filename in required_files.items():
    found = file_exists(project_path, filename)
    required_status.append(
        {
            "Item": label,
            "File": filename,
            "Status": "Found" if found else "Missing",
        }
    )

for label, filename in optional_files.items():
    found = file_exists(project_path, filename)
    optional_status.append(
        {
            "Item": label,
            "File": filename,
            "Status": "Found" if found else "Not found",
        }
    )

st.markdown("#### Required for Report")

required_df = pd.DataFrame(required_status)
st.dataframe(required_df, use_container_width=True, hide_index=True)

st.markdown("#### Optional Supporting Files")

optional_df = pd.DataFrame(optional_status)
st.dataframe(optional_df, use_container_width=True, hide_index=True)

required_ready = all(item["Status"] == "Found" for item in required_status)

if required_ready:
    st.success("All required report inputs are available.")
else:
    st.warning(
        "Some required report inputs are missing. Complete Kinematics analysis before generating the report."
    )


# -----------------------------------------------------------------------------
# Preview statistics
# -----------------------------------------------------------------------------

if file_exists(project_path, "statistics.csv"):
    st.divider()
    st.markdown("### Statistics Preview")

    try:
        stats_df = pd.read_csv(project_path / "statistics.csv")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.warning(f"Could not preview statistics.csv: {exc}")


# -----------------------------------------------------------------------------
# Preview figures
# -----------------------------------------------------------------------------

figure_files = [
    ("Clinical Joint Angles Over Time", project_path / "figures" / "kinematics_time_curves.png"),
    ("Normalized Clinical Joint Angles", project_path / "figures" / "kinematics_normalized_curves.png"),
]

existing_figures = [
    (title, path)
    for title, path in figure_files
    if path.exists()
]

if existing_figures:
    st.divider()
    st.markdown("### Figure Preview")

    for title, path in existing_figures:
        st.markdown(f"#### {title}")
        st.image(str(path), use_container_width=True)


# -----------------------------------------------------------------------------
# Generate report
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Generate PDF Report")

st.write(
    "The PDF report will include session metadata, recording information, processing steps, "
    "clinical angle convention, limitation statement, kinematics statistics, and figures."
)

generate_button = st.button(
    "Generate PDF Report",
    type="primary",
    use_container_width=True,
    disabled=not required_ready,
)

if generate_button:
    try:
        report_info = generate_report(project_path)

        metadata = load_metadata(project_path)

        if metadata is not None:
            metadata["current_step"] = "report"
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "processing" not in metadata:
                metadata["processing"] = {}

            metadata["processing"]["report"] = report_info

            if "files" not in metadata:
                metadata["files"] = {}

            metadata["files"]["report_pdf"] = "report/report.pdf"
            metadata["files"]["report_info"] = "report/report_info.json"

            save_metadata(project_path, metadata)

        st.success("PDF report generated successfully.")
        st.rerun()

    except Exception as exc:
        st.error(f"Report generation failed: {exc}")


# -----------------------------------------------------------------------------
# Report output
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Report Output Files")

report_pdf_exists = file_exists(project_path, "report/report.pdf")
report_info_exists = file_exists(project_path, "report/report_info.json")

if report_pdf_exists:
    st.success(f"PDF report: `report/{REPORT_FILENAME}` found")

    report_path = project_path / "report" / REPORT_FILENAME

    with open(report_path, "rb") as pdf_file:
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name=f"{project_path.name}_VMotionLab_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    st.warning(f"PDF report: `report/{REPORT_FILENAME}` not found yet")

if report_info_exists:
    st.success(f"Report information: `report/{REPORT_INFO_FILENAME}` found")
else:
    st.info(f"Report information: `report/{REPORT_INFO_FILENAME}` not found yet")


if st.button("Open Project Folder", use_container_width=True):
    success = open_project_folder(project_path)

    if success:
        st.toast("Project folder opened.")
    else:
        st.error("Could not open project folder.")


st.divider()
st.markdown("### Workflow Status")
show_workflow_status(project_path)

# -----------------------------------------------------------------------------
# Finish workflow
# -----------------------------------------------------------------------------

st.divider()
st.markdown("### Finish")

st.success(
    "Analysis workflow completed. Click Done to return to the Home page."
)

col_back, col_done = st.columns([1, 1])

with col_back:
    if st.button("⬅ Back to Kinematics", use_container_width=True):
        st.switch_page("pages/5_Kinematics.py")

with col_done:
    if st.button("Done", type="primary", use_container_width=True):
        st.switch_page("Home.py")