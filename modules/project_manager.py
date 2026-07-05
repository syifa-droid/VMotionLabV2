from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
CURRENT_PROJECT_FILE = DATA_DIR / "current_project.json"


STEP_LABELS = {
    "upload_video": "Upload Videos",
    "preprocessing": "Preprocessing",
    "kinematics": "Kinematics",
    "report": "Report",
}


STEP_FILES = {
    "upload_video": ["motion_raw.csv"],
    "preprocessing": ["motion_filtered.csv"],
    "kinematics": ["statistics.csv", "normalized_curves.csv"],
    "report": ["report/report.pdf"],
}


def ensure_data_folders() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    value = str(value).strip()

    if not value:
        value = "Unknown"

    value = re.sub(r"[^\w\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value or "Unknown"


def build_project_folder_name(
    subject_id: str,
    task: str,
    trial_name: str,
) -> str:
    subject_id = sanitize_name(subject_id)
    task = sanitize_name(task)
    trial_name = sanitize_name(trial_name)

    return f"{subject_id}_{task}_{trial_name}"


def get_project_path(
    subject_id: str,
    task: str,
    trial_name: str,
) -> Path:
    ensure_data_folders()

    folder_name = build_project_folder_name(
        subject_id=subject_id,
        task=task,
        trial_name=trial_name,
    )

    return PROJECTS_DIR / folder_name


def save_json(path: Path | str, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path | str) -> Optional[Dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_metadata(project_path: Path | str, metadata: Dict[str, Any]) -> None:
    project_path = Path(project_path)
    project_path.mkdir(parents=True, exist_ok=True)

    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

    save_json(project_path / "metadata.json", metadata)


def load_metadata(project_path: Path | str) -> Optional[Dict[str, Any]]:
    project_path = Path(project_path)
    return load_json(project_path / "metadata.json")


def create_project(
    subject_id: str,
    task: str,
    trial_name: str,
    side: str = "",
    clinician: str = "",
    notes: str = "",
    source_video_name: str = "",
) -> Dict[str, Any]:
    """
    Create one trial project folder.

    In simplified VMotionLabV2:
    - one uploaded video = one trial project
    """
    ensure_data_folders()

    project_path = get_project_path(
        subject_id=subject_id,
        task=task,
        trial_name=trial_name,
    )

    project_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now().isoformat(timespec="seconds")

    metadata = {
        "app": "VMotionLabV2",
        "workflow": "upload_video_only",
        "project_path": str(project_path),
        "project_name": project_path.name,
        "subject_id": str(subject_id).strip(),
        "task": str(task).strip(),
        "trial_name": str(trial_name).strip(),
        "side": str(side).strip(),
        "clinician": str(clinician).strip(),
        "notes": str(notes).strip(),
        "source_video_name": str(source_video_name).strip(),
        "current_step": "upload_video",
        "created_at": now,
        "updated_at": now,
        "files": {},
        "processing": {},
        "kinematics": {},
        "report": {},
    }

    save_metadata(project_path, metadata)

    return {
        "project_path": str(project_path),
        "metadata": metadata,
    }


def set_current_project(project_path: Path | str) -> None:
    ensure_data_folders()

    project_path = Path(project_path)

    save_json(
        CURRENT_PROJECT_FILE,
        {
            "active_project_path": str(project_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def get_current_project() -> Optional[Path]:
    data = load_json(CURRENT_PROJECT_FILE)

    if not data:
        return None

    project_path = data.get("active_project_path")

    if not project_path:
        return None

    project_path = Path(project_path)

    if not project_path.exists():
        return None

    return project_path


def clear_current_project() -> None:
    if CURRENT_PROJECT_FILE.exists():
        CURRENT_PROJECT_FILE.unlink()


def file_exists(project_path: Path | str, relative_path: str) -> bool:
    project_path = Path(project_path)
    return (project_path / relative_path).exists()


def list_project_paths() -> List[Path]:
    ensure_data_folders()

    project_paths = []

    for path in PROJECTS_DIR.iterdir():
        if path.is_dir() and (path / "metadata.json").exists():
            project_paths.append(path)

    return sorted(
        project_paths,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def list_project_metadata() -> List[Dict[str, Any]]:
    projects = []

    for project_path in list_project_paths():
        metadata = load_metadata(project_path)

        if metadata is None:
            continue

        metadata["project_path"] = str(project_path)
        metadata["project_name"] = project_path.name

        projects.append(metadata)

    return projects


def is_project_step_complete(
    project_path: Path | str,
    step: str,
) -> bool:
    required_files = STEP_FILES.get(step, [])

    if not required_files:
        return False

    return all(file_exists(project_path, filename) for filename in required_files)


def find_completed_projects(
    required_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if required_files is None:
        required_files = ["motion_raw.csv"]

    completed = []

    for metadata in list_project_metadata():
        project_path = Path(metadata["project_path"])

        if all((project_path / filename).exists() for filename in required_files):
            completed.append(metadata)

    return completed


def update_project_files(
    project_path: Path | str,
    files: Dict[str, str],
) -> None:
    metadata = load_metadata(project_path)

    if metadata is None:
        return

    if "files" not in metadata:
        metadata["files"] = {}

    metadata["files"].update(files)

    save_metadata(project_path, metadata)


def update_current_step(
    project_path: Path | str,
    current_step: str,
) -> None:
    metadata = load_metadata(project_path)

    if metadata is None:
        return

    metadata["current_step"] = current_step

    save_metadata(project_path, metadata)

# -----------------------------------------------------------------------------
# Backward compatibility helpers
# -----------------------------------------------------------------------------
# Some existing VMotionLabV2 files still use older function names.
# Keep these aliases so Home.py, state.py, and other pages continue to work.


def ensure_directories() -> None:
    """Backward-compatible alias for ensure_data_folders()."""
    ensure_data_folders()


def set_active_project(project_path: Path | str) -> None:
    """Backward-compatible alias for set_current_project()."""
    set_current_project(project_path)


def get_active_project() -> Optional[Path]:
    """Backward-compatible alias for get_current_project()."""
    return get_current_project()


def get_active_project_path() -> Optional[Path]:
    """Backward-compatible alias for get_current_project()."""
    return get_current_project()


def clear_active_project() -> None:
    """Backward-compatible alias for clear_current_project()."""
    clear_current_project()

def get_next_incomplete_step(project_path: Path | str) -> str:
    """
    Return the next incomplete workflow step.

    Simplified VMotionLabV2 workflow:
    Upload Videos -> Preprocessing -> Kinematics -> Report
    """
    project_path = Path(project_path)

    workflow_order = [
        "upload_video",
        "preprocessing",
        "kinematics",
        "report",
    ]

    for step in workflow_order:
        if not is_project_step_complete(project_path, step):
            return step

    return "report"

    for step in workflow_order:
        if not is_project_step_complete(project_path, step):
            return step

    return "report"

def set_current_projects(project_paths: list[Path | str]) -> None:
    """
    Save one or more selected active trial paths.
    The first selected project is also saved as the single active project
    for backward compatibility.
    """
    ensure_data_folders()

    clean_paths = [str(Path(path)) for path in project_paths]

    save_json(
        CURRENT_PROJECT_FILE,
        {
            "active_project_path": clean_paths[0] if clean_paths else "",
            "active_project_paths": clean_paths,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

def get_current_projects() -> list[Path]:
    """
    Return selected active trial paths.
    Falls back to single active project if old current_project.json is used.
    """
    data = load_json(CURRENT_PROJECT_FILE)

    if not data:
        return []

    paths = data.get("active_project_paths", [])

    if not paths:
        single_path = data.get("active_project_path")

        if single_path:
            paths = [single_path]

    valid_paths = []

    for path in paths:
        project_path = Path(path)

        if project_path.exists():
            valid_paths.append(project_path)

    return valid_paths

def open_project_folder(project_path: Path | str) -> Path:
    """
    Backward-compatible helper.

    In the simplified public VMotionLabV2 version, we do not automatically
    open File Explorer. This function only returns the project folder path.
    """
    project_path = Path(project_path)
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path