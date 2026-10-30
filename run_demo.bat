@echo off
python scripts\generate_demo_data.py
if errorlevel 1 exit /b 1
streamlit run app.py
