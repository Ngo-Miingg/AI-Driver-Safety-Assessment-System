@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "run_mmrt_v2_demo.py"
) else (
  python "run_mmrt_v2_demo.py"
)
