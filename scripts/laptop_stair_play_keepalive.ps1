param(
  [Parameter(Mandatory = $true)]
  [string]$RunDir,
  [Parameter(Mandatory = $true)]
  [string]$Checkpoint,
  [Parameter(Mandatory = $true)]
  [string]$Repo,
  [Parameter(Mandatory = $true)]
  [string]$CacheRoot,
  [Parameter(Mandatory = $true)]
  [string]$TempRoot,
  [Parameter(Mandatory = $true)]
  [string]$UvCacheDir,
  [Parameter(Mandatory = $true)]
  [string]$UvPythonInstallDir,
  [ValidateRange(0, 9)]
  [int]$TerrainLevel = 1
)

$ErrorActionPreference = 'Continue'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$CheckpointPath = Join-Path $Repo "logs\remote_watch\$RunDir\$Checkpoint"
$LogDir = Join-Path $Repo "logs\remote_watch\$RunDir"
$Out = Join-Path $LogDir 'laptop_viser_keepalive.out.log'
$Err = Join-Path $LogDir 'laptop_viser_keepalive.err.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

$env:UV_CACHE_DIR = $UvCacheDir
$env:UV_PYTHON_INSTALL_DIR = $UvPythonInstallDir
$env:XDG_CACHE_HOME = $CacheRoot
$env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot 'pycache'
$env:MPLCONFIGDIR = Join-Path $CacheRoot 'matplotlib'
$env:RERUN_CACHE_DIR = Join-Path $CacheRoot 'rerun'
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:SE3_WATCH_TERRAIN_LEVEL = "$TerrainLevel"
$env:SE3_TRAIN_VIEW_TERRAIN_LEVEL = "$TerrainLevel"
$checkpointMatch = [regex]::Match($Checkpoint, '^model_(\d+)\.pt$')
$checkpointIter = if ($checkpointMatch.Success) { $checkpointMatch.Groups[1].Value } else { '0' }
$env:SE3_WATCH_ITER = $checkpointIter
$env:SE3_TRAIN_VIEW_ITER = $checkpointIter

while ($true) {
  Add-Content -LiteralPath $Out -Value "$(Get-Date -Format o) start $CheckpointPath"
  Set-Location $Repo
  & $Python -u -m se3_sim2sim.cli --checkpoint $CheckpointPath --model-variant closedchain --viewer viser --device cpu --print-every 0 --stair-terrain --stair-terrain-level $TerrainLevel --command 1.2 0 0 0 0.32 0 0 0 1>> $Out 2>> $Err
  Add-Content -LiteralPath $Out -Value "$(Get-Date -Format o) exited code=$LASTEXITCODE"
  Start-Sleep -Seconds 3
}
