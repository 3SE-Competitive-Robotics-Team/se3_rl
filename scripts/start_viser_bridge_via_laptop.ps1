param(
    [ValidateSet("abbtask", "llm")]
    [string]$TargetKind = "abbtask",
    [int]$LocalPort = 18198,
    [string]$LaptopHost = "laptop-imgpi2nm-shanghai",
    [string]$RemoteSshHost = "",
    [string]$Target = "",
    [string]$LaptopScriptPath = "E:/se3_stair_viewer_setup/laptop_viser_bridge.ps1"
)

$ErrorActionPreference = "Stop"

if (-not $RemoteSshHost) {
    $RemoteSshHost = if ($TargetKind -eq "llm") { "llm" } else { "a800" }
}

if (-not $Target) {
    $Target = if ($TargetKind -eq "llm") { "127.0.0.1:8080" } else { "172.16.6.130:8080" }
}

$tunnelPort = $LocalPort + 1
$localScript = Join-Path $PSScriptRoot "laptop_viser_bridge.ps1"
if (-not (Test-Path -LiteralPath $localScript)) {
    throw "找不到 laptop bridge 脚本：$localScript"
}

Write-Host "上传 laptop bridge 脚本到 $LaptopHost..."
scp -o BatchMode=yes $localScript "${LaptopHost}:$LaptopScriptPath"
if ($LASTEXITCODE -ne 0) {
    throw "上传 laptop bridge 脚本失败。"
}

$remoteWindowsPath = $LaptopScriptPath -replace "/", "\"

Write-Host "启动 Viser bridge："
Write-Host "  local        http://127.0.0.1:$LocalPort"
Write-Host "  laptop       $LaptopHost"
Write-Host "  remote ssh   $RemoteSshHost"
Write-Host "  target       $Target"
Write-Host "保持此窗口打开；Ctrl+C 会关闭 bridge。"

ssh `
    -o BatchMode=yes `
    -o ExitOnForwardFailure=yes `
    -L "127.0.0.1:$LocalPort`:127.0.0.1:$LocalPort" `
    $LaptopHost `
    powershell -NoProfile -ExecutionPolicy Bypass -File $remoteWindowsPath `
    -BridgePort $LocalPort `
    -TunnelPort $tunnelPort `
    -RemoteSshHost $RemoteSshHost `
    -Target $Target
