@echo off
set RUN_DIR=2026-06-16_07-45-39_stair_holdradial_m4999_8gpu_4096pergpu_20260616_1543
set CHECKPOINT=model_0.pt
set REPO=E:\se3_stair_viewer
set LOG_DIR=%REPO%\logs\remote_watch\%RUN_DIR%
set CKPT=%LOG_DIR%\%CHECKPOINT%
set OUT=%LOG_DIR%\laptop_viser_keepalive_cmd.out.log
set ERR=%LOG_DIR%\laptop_viser_keepalive_cmd.err.log
set CACHE_ROOT=E:\se3_stair_viewer_setup\cache
set TEMP_ROOT=E:\se3_stair_viewer_setup\tmp
set UV_CACHE_DIR=E:\uv-cache
set UV_PYTHON_INSTALL_DIR=E:\uv-python
set XDG_CACHE_HOME=%CACHE_ROOT%
set PYTHONPYCACHEPREFIX=%CACHE_ROOT%\pycache
set MPLCONFIGDIR=%CACHE_ROOT%\matplotlib
set RERUN_CACHE_DIR=%CACHE_ROOT%\rerun
set TEMP=%TEMP_ROOT%
set TMP=%TEMP_ROOT%
set SE3_WATCH_TERRAIN_LEVEL=1
set SE3_TRAIN_VIEW_TERRAIN_LEVEL=1
set SE3_WATCH_ITER=0
set SE3_TRAIN_VIEW_ITER=0
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%CACHE_ROOT%" mkdir "%CACHE_ROOT%"
if not exist "%TEMP_ROOT%" mkdir "%TEMP_ROOT%"
cd /d "%REPO%"
:loop
echo %DATE% %TIME% start "%CKPT%" >> "%OUT%"
.venv\Scripts\python.exe -u -m se3_sim2sim.cli --checkpoint "%CKPT%" --model-variant closedchain --viewer viser --device cpu --print-every 0 --stair-terrain --stair-terrain-level 1 --command 1.2 0 0 0 0.32 0 0 0 >> "%OUT%" 2>> "%ERR%"
echo %DATE% %TIME% exited code=%ERRORLEVEL% >> "%OUT%"
timeout /t 3 /nobreak >nul
goto loop
