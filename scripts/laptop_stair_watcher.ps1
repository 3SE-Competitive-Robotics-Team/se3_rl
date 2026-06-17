param(
  [switch]$Once
)

$ErrorActionPreference = 'Continue'
$Namespace = 'gczx-project06'
$Pod = 'abbtask-79cdb78487-mgx44'
$Project = '/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg'
$RunDir = '2026-06-17_07-28-26_stair_strict_success_a3887a5_8gpu_4096pergpu_20260617'
$Task = 'SE3-WheelLegged-Stair-GRU'
$TrainLog = '/tmp/stair_strict_formal_a3887a5.log'
$Repo = 'E:\se3_stair_viewer'
$LocalRunDir = Join-Path $Repo "logs\remote_watch\$RunDir"
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$CacheRoot = 'E:\se3_stair_viewer_setup\cache'
$TempRoot = 'E:\se3_stair_viewer_setup\tmp'
$IntervalIters = 100
$PollSeconds = 60
$StepHeightMin = 0.05
$StepHeightMax = 0.20
$MoveUpMinSteps = 2.0
$LastIter = -1
$LastTerrainLevel = -1
$Viewer = $null
New-Item -ItemType Directory -Force -Path $LocalRunDir | Out-Null
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

function Invoke-A800HostBash([string]$ScriptText) {
  $normalized = $ScriptText -replace "`r`n", "`n"
  $hostB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
  & ssh -T -o BatchMode=yes -o ConnectTimeout=10 a800 "echo $hostB64 | base64 -d | bash"
  if ($LASTEXITCODE -ne 0) { throw "a800 host command failed: $LASTEXITCODE" }
}

function Invoke-PodBash([string]$ScriptText, [int]$TimeoutSeconds = 60) {
  $normalized = $ScriptText -replace "`r`n", "`n"
  $podB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
  $hostScript = @"
set -euo pipefail
timeout ${TimeoutSeconds}s kubectl exec -n $Namespace $Pod -- bash -lc 'echo $podB64 | base64 -d | bash'
"@
  Invoke-A800HostBash $hostScript
}

function Get-LatestCheckpoint() {
  $podScript = @"
set -euo pipefail
cd '$Project'
for f in logs/rsl_rl/se3_wheel_leg/$RunDir/model_*.pt; do
  [ -f "`$f" ] || continue
  base="`$(basename "`$f")"
  [[ "`$base" =~ ^model_[0-9]+\.pt$ ]] || continue
  printf '%s\t%s\n' "`$base" "`$(stat -c%s "`$f")"
done | sort -V | tail -1
"@
  $line = (
    Invoke-PodBash $podScript 60 |
      Where-Object { $_ -match '^model_\d+\.pt\s+\d+' } |
      Select-Object -Last 1
  )
  if (-not $line) { return $null }
  if ($line -notmatch '^model_(\d+)\.pt\s+(\d+)') { throw "unexpected checkpoint line: $line" }
  [pscustomobject]@{ Name = "model_$($Matches[1]).pt"; Iter = [int]$Matches[1]; Size = [int64]$Matches[2] }
}

function Copy-Checkpoint($ckpt) {
  $localPath = Join-Path $LocalRunDir $ckpt.Name
  if ((Test-Path $localPath) -and ((Get-Item $localPath).Length -eq $ckpt.Size)) { return $localPath }
  $safeRun = $RunDir -replace '[^A-Za-z0-9_.-]', '_'
  $hostTmp = "/tmp/se3_laptop_watch_${safeRun}_$($ckpt.Name)"
  $remotePath = "$Project/logs/rsl_rl/se3_wheel_leg/$RunDir/$($ckpt.Name)"
  $hostScript = @"
set -euo pipefail
rm -f '$hostTmp'
timeout 180s kubectl cp -n $Namespace '${Pod}:$remotePath' '$hostTmp'
stat -c%s '$hostTmp'
"@
  $sizeLine = (Invoke-A800HostBash $hostScript | Select-Object -Last 1)
  if ([int64]$sizeLine -ne $ckpt.Size) { throw "host tmp size mismatch: $sizeLine != $($ckpt.Size)" }
  $tmpLocal = "$localPath.tmp"
  Remove-Item -LiteralPath $tmpLocal -Force -ErrorAction SilentlyContinue
  & scp -o BatchMode=yes -o ConnectTimeout=10 "a800:$hostTmp" $tmpLocal
  if ($LASTEXITCODE -ne 0) { throw "scp failed: $LASTEXITCODE" }
  if ((Get-Item $tmpLocal).Length -ne $ckpt.Size) { throw "local size mismatch for $($ckpt.Name)" }
  Move-Item -LiteralPath $tmpLocal -Destination $localPath -Force
  Invoke-A800HostBash "rm -f '$hostTmp'" | Out-Null
  $localPath
}

function Get-WatchTerrainLevel() {
  $podScript = @"
set -euo pipefail
if [ -f '$TrainLog' ]; then
  grep -E 'Curriculum/terrain_levels/(move_up_height_mean|level_mean)' '$TrainLog' | tail -20
fi
"@
  try {
  $lines = @(Invoke-PodBash $podScript 20)
  } catch {
    return 0
  }
  [array]::Reverse($lines)
  foreach ($line in $lines) {
    if ($line -match 'move_up_height_mean:\s*([-+0-9.eE]+)') {
      $moveUpHeight = 0.0
      if ([double]::TryParse($Matches[1], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$moveUpHeight)) {
        $stepHeight = $moveUpHeight / [math]::Max(1.0e-6, $MoveUpMinSteps)
        $alpha = ($stepHeight - $StepHeightMin) / [math]::Max(1.0e-6, $StepHeightMax - $StepHeightMin)
        $level = [int][math]::Round($alpha * 9.0)
        return [math]::Max(0, [math]::Min(9, $level))
      }
    }
  }
  foreach ($line in $lines) {
    if ($line -match 'level_mean:\s*([-+0-9.eE]+)') {
      $levelMean = 0.0
      if ([double]::TryParse($Matches[1], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$levelMean)) {
        $level = [int][math]::Round($levelMean)
        return [math]::Max(0, [math]::Min(9, $level))
      }
    }
  }
  0
}

function Stop-Viewer() {
  if ($script:Viewer -and -not $script:Viewer.HasExited) {
    Stop-Process -Id $script:Viewer.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -match 'se3_sim2sim\.cli' -and
      $_.CommandLine -match [regex]::Escape($Repo)
    } |
    ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Start-Viewer($checkpointPath, [int]$terrainLevel) {
  Stop-Viewer
  $out = Join-Path $LocalRunDir 'viser_play.out.log'
  $err = Join-Path $LocalRunDir 'viser_play.err.log'
  $env:SE3_WATCH_TERRAIN_LEVEL = "$terrainLevel"
  $env:SE3_TRAIN_VIEW_TERRAIN_LEVEL = "$terrainLevel"
  $env:SE3_WATCH_ITER = ([regex]::Match((Split-Path $checkpointPath -Leaf), 'model_(\d+)\.pt').Groups[1].Value)
  $env:SE3_TRAIN_VIEW_ITER = $env:SE3_WATCH_ITER
  $args = @('-m','se3_sim2sim.cli','--checkpoint',$checkpointPath,'--model-variant','closedchain','--viewer','viser','--device','cpu','--print-every','0','--stair-terrain','--stair-terrain-level',"$terrainLevel",'--stair-ctbc','--stair-ctbc-iter',$env:SE3_WATCH_ITER,'--command','1.2','0','0','0','0.32','0','0','0')
  $script:Viewer = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Repo -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
  Write-Output "viewer pid=$($script:Viewer.Id) terrain_level=$terrainLevel checkpoint=$checkpointPath"
}

try {
  while ($true) {
    try {
      $ckpt = Get-LatestCheckpoint
      if ($ckpt) {
        $terrainLevel = Get-WatchTerrainLevel
        $terrainChanged = ($LastTerrainLevel -lt 0) -or ($terrainLevel -ne $LastTerrainLevel)
        $needLaunch = ($LastIter -lt 0) -or (($ckpt.Iter - $LastIter) -ge $IntervalIters) -or ($Viewer -and $Viewer.HasExited) -or $terrainChanged
        if ($needLaunch) {
          Write-Output "latest $($ckpt.Name) iter=$($ckpt.Iter) size=$($ckpt.Size)"
          $path = Copy-Checkpoint $ckpt
          Write-Output "copied $path"
          Start-Viewer $path $terrainLevel
          $LastIter = $ckpt.Iter
          $LastTerrainLevel = $terrainLevel
        }
      } else {
        Write-Output 'no checkpoint yet'
      }
    } catch {
      Write-Output "watch error: $($_.Exception.Message)"
    }
    if ($Once) { break }
    Start-Sleep -Seconds $PollSeconds
  }
} finally {
  if ($Once) { Stop-Viewer }
}
