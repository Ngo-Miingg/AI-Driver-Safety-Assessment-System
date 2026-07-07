@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY_LOCAL=%ROOT%.venv\Scripts\python.exe"
set "VENV_PY_WORK=%ROOT%..\..\..\.venv\Scripts\python.exe"

if exist "%VENV_PY_LOCAL%" (
  "%VENV_PY_LOCAL%" "%ROOT%run_mddad_demo_7class.py"
) else if exist "%VENV_PY_WORK%" (
  "%VENV_PY_WORK%" "%ROOT%run_mddad_demo_7class.py"
) else (
  python "%ROOT%run_mddad_demo_7class.py"
)
