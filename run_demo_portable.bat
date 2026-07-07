@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv\Scripts\python.exe
  echo Hay dam bao bundle nay co kem thu muc .venv.
  exit /b 1
)

echo [1/3] Kill old demo processes...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'run_mddad_demo_geom2.py' -or $_.CommandLine -match 'run_mddad_demo_7class.py' -or $_.CommandLine -match 'run_model_test_simple.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

echo [2/3] Start GEOM2 demo on http://127.0.0.1:5002 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '.venv\\Scripts\\python.exe' -ArgumentList 'run_mddad_demo_geom2.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden"

echo [3/3] Start FINAL demo on http://127.0.0.1:5003 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '.venv\\Scripts\\python.exe' -ArgumentList 'run_mddad_demo_7class.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden"

timeout /t 6 /nobreak >nul

echo.
echo Demo URLs:
echo   GEOM2 : http://127.0.0.1:5002/model-test-simple-cam
echo   FINAL : http://127.0.0.1:5003/model-test-simple-cam
echo.
echo Login: admin / admin123

endlocal
