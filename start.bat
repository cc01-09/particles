@echo off
REM One-command start (Windows).
REM Creates .venv on first run, installs dependencies, then launches run.py.
REM Usage: start.bat [--backend cpu|hailo|mock] [--camera N] [...]
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  REM MediaPipe 0.10.21 ships wheels for Python 3.9 - 3.12; prefer 3.12 / 3.11.
  set "PY="
  py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
  if not defined PY ( py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11" )
  if not defined PY (
    echo Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/
    exit /b 1
  )
  echo Creating virtual environment...
  %PY% -m venv .venv || exit /b 1
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
)

".venv\Scripts\python.exe" run.py %*
