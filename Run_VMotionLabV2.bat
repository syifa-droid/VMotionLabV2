@echo off
cd /d C:\Users\SF\VMotionLabV2
call .venv\Scripts\activate
python -m streamlit run Home.py
pause