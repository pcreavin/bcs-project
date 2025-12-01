@echo off
REM Helper batch file to activate the virtual environment in Command Prompt (Windows)
if not exist venv\Scripts\activate.bat (
  echo Virtual environment not found. Create one with:
  echo   python -m venv venv
  exit /b 1
)
call venv\Scripts\activate.bat
echo Activated virtual environment (Command Prompt). Installed packages are isolated to the venv.
echo To install dependencies: pip install -r requirements.txt
