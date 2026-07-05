from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from modules.project_manager import (
    ensure_directories,
    get_current_project,
    get_current_projects,
    set_current_project,
    set_current_projects,
)


def initialize_state() -> None:
    """
    Initialize VMotionLabV2 session state.
    """
    ensure_directories()

    if "active_project_path" not in st.session_state:
        st.session_state["active_project_path"] = None

    if "active_project_paths" not in st.session_state:
        st.session_state["active_project_paths"] = []

    restore_active_project()


def restore_active_project() -> None:
    """
    Restore active trial(s) from data/current_project.json.
    Compatible with both single-trial and multi-trial selection.
    """
    current_projects = get_current_projects()

    if current_projects:
        st.session_state["active_project_paths"] = [
            str(path) for path in current_projects
        ]

        st.session_state["active_project_path"] = str(current_projects[0])
        return

    current_project = get_current_project()

    if current_project is not None:
        st.session_state["active_project_path"] = str(current_project)
        st.session_state["active_project_paths"] = [str(current_project)]


def set_active_project_path(project_path: Path | str) -> None:
    """
    Set one active trial.
    """
    project_path = Path(project_path)

    set_current_project(project_path)

    st.session_state["active_project_path"] = str(project_path)
    st.session_state["active_project_paths"] = [str(project_path)]


def set_active_project_paths(project_paths: list[Path | str]) -> None:
    """
    Set one or more active trials.
    """
    clean_paths = [Path(path) for path in project_paths]

    set_current_projects(clean_paths)

    st.session_state["active_project_paths"] = [
        str(path) for path in clean_paths
    ]

    if clean_paths:
        st.session_state["active_project_path"] = str(clean_paths[0])
    else:
        st.session_state["active_project_path"] = None


def get_active_project_path() -> Optional[Path]:
    """
    Get primary active trial path.
    """
    active_project_path = st.session_state.get("active_project_path")

    if not active_project_path:
        return None

    project_path = Path(active_project_path)

    if not project_path.exists():
        return None

    return project_path


def get_active_project_paths() -> list[Path]:
    """
    Get all selected active trial paths.
    """
    active_project_paths = st.session_state.get("active_project_paths", [])

    valid_paths = []

    for path in active_project_paths:
        project_path = Path(path)

        if project_path.exists():
            valid_paths.append(project_path)

    return valid_paths


def require_active_project() -> Path:
    """
    Require one active trial.
    """
    active_project_path = get_active_project_path()

    if active_project_path is None:
        st.warning("No active trial selected. Please upload or select a trial first.")

        if st.button("Go to Upload Videos", type="primary"):
            st.switch_page("pages/2_Upload_Video.py")

        st.stop()

    return active_project_path


def require_active_projects() -> list[Path]:
    """
    Require one or more active trials.
    """
    active_project_paths = get_active_project_paths()

    if not active_project_paths:
        st.warning("No active trials selected. Please upload or select trial(s) first.")

        if st.button("Go to Upload Videos", type="primary"):
            st.switch_page("pages/2_Upload_Video.py")

        st.stop()

    return active_project_paths


def clear_active_project_state() -> None:
    """
    Clear active trial session state only.
    """
    st.session_state["active_project_path"] = None
    st.session_state["active_project_paths"] = []