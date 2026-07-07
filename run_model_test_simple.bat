@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%..\..\..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%ROOT%run_model_test_simple.py"
) else (
  python "%ROOT%run_model_test_simple.py"
)
