$ErrorActionPreference = 'Continue'
$RunDir = '2026-06-16_07-45-39_stair_holdradial_m4999_8gpu_4096pergpu_20260616_1543'
$Checkpoint = 'model_0.pt'
$Repo = 'E:\se3_stair_viewer'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$CheckpointPath = Join-Path $Repo "logs\remote_watch\$RunDir\$Checkpoint"
$LogDir = Join-Path $Repo "logs\remote_watch\$RunDir"
$Out = Join-Path $LogDir 'laptop_viser_keepalive.out.log'
$Err = Join-Path $LogDir 'laptop_viser_keepalive.err.log'
$CacheRoot = 'E:\se3_stair_viewer_setup\cache'
$TempRoot = 'E:\se3_stair_viewer_setup\tmp'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

$env:UV_CACHE_DIR = 'E:\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = 'E:\uv-python'
$env:XDG_CACHE_HOME = $CacheRoot
$env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot 'pycache'
$env:MPLCONFIGDIR = Join-Path $CacheRoot 'matplotlib'
$env:RERUN_CACHE_DIR = Join-Path $CacheRoot 'rerun'
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:SE3_WATCH_TERRAIN_LEVEL = '1'
$env:SE3_TRAIN_VIEW_TERRAIN_LEVEL = '1'
$env:SE3_WATCH_ITER = '0'
$env:SE3_TRAIN_VIEW_ITER = '0'

while ($true) {
  Add-Content -LiteralPath $Out -Value "$(Get-Date -Format o) start $CheckpointPath"
  Set-Location $Repo
  & $Python -u -m se3_sim2sim.cli --checkpoint $CheckpointPath --model-variant closedchain --viewer viser --device cpu --print-every 0 --stair-terrain --stair-terrain-level 1 --command 1.2 0 0 0 0.32 0 0 0 1>> $Out 2>> $Err
  Add-Content -LiteralPath $Out -Value "$(Get-Date -Format o) exited code=$LASTEXITCODE"
  Start-Sleep -Seconds 3
}
