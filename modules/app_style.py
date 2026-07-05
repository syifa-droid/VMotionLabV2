"""
VMotionLab - Global App Style
=============================

Global styling for Streamlit pages.

App display font:
- Poppins for page content

PDF report font:
- Keep Arial in report_generator.py
"""

from __future__ import annotations

import streamlit as st


APP_FONT_FAMILY = "'Poppins', Arial, Helvetica, sans-serif"


def apply_global_style() -> None:
    """
    Apply global Poppins font style to VMotionLab content without breaking
    Streamlit internal icons such as the sidebar collapse button.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

        /* Main readable content */
        h1, h2, h3, h4, h5, h6,
        p,
        label,
        input,
        textarea,
        select,
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown span,
        .stMarkdown div,
        .stCaption,
        .stAlert,
        .stSelectbox,
        .stMultiSelect,
        .stTextInput,
        .stTextArea,
        .stNumberInput,
        .stRadio,
        .stCheckbox,
        .stMetric,
        .stDataFrame {{
            font-family: {APP_FONT_FAMILY} !important;
        }}

        /* Sidebar text only, not icons */
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {{
            font-family: {APP_FONT_FAMILY} !important;
        }}

        /* Normal app buttons, but avoid Streamlit header/sidebar control buttons */
        div[data-testid="stButton"] button {{
            font-family: {APP_FONT_FAMILY} !important;
        }}

        /* Home title classes */
        .vmotion-title {{
            font-family: {APP_FONT_FAMILY} !important;
            font-size: 32px;
            font-weight: 700;
            color: #00897B;
            margin-bottom: 0.25rem;
        }}

        .vmotion-subtitle {{
            font-family: {APP_FONT_FAMILY} !important;
            font-size: 18px;
            font-weight: 500;
            color: #37474F;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )