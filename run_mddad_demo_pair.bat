@echo off
setlocal

cd /d D:\Work\HocKy3\ComputerVision\MDDAD

echo [1/3] Kill old demo processes...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'run_mddad_demo_geom2.py' -or $_.CommandLine -match 'run_mddad_demo_7class.py' -or $_.CommandLine -match 'run_model_test_simple.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

echo [2/3] Start GEOM2 demo on http://127.0.0.1:5002 ...
start "MDDAD GEOM2" /min D:\Work\.venv\Scripts\python.exe run_mddad_demo_geom2.py

echo [3/3] Start FINAL demo on http://127.0.0.1:5003 ...
start "MDDAD FINAL" /min D:\Work\.venv\Scripts\python.exe run_mddad_demo_7class.py

timeout /t 6 /nobreak >nul

echo.
echo Demo URLs:
echo   GEOM2 : http://127.0.0.1:5002/model-test-simple-cam
echo   FINAL : http://127.0.0.1:5003/model-test-simple-cam
echo.
echo IMPORTANT:
echo   - Do not use the old 5001 page.
echo   - Log in with admin / admin123 first.
echo   - Then start realtime on 5002 or 5003.

endlocal
