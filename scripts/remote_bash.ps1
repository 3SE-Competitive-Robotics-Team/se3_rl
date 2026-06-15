<#
.SYNOPSIS
通过 SSH 在远端 bash 中执行脚本，避免 Windows PowerShell 提前解析 bash 语法。

.DESCRIPTION
本脚本把用户提供的 bash 脚本编码成 UTF-8 base64，再通过一个极小的
远端 bootstrap 解码执行。PowerShell 侧不再直接承载 `&&`、`$()`、管道
或多层引号，因此适合从 Windows PowerShell 调用远程训练机或 Kubernetes pod。

.EXAMPLE
$bash = @'
set -x
date
grep -R "reward" logs | tail -20
'@
.\scripts\remote_bash.ps1 -HostAlias wuyinyun -ScriptText $bash -UseProxy

.EXAMPLE
.\scripts\remote_bash.ps1 -HostAlias wuyinyun -ScriptPath .\tmp\check_training.sh
#>

[CmdletBinding(DefaultParameterSetName = "Text")]
param(
    [string]$HostAlias = "wuyinyun",

    [Parameter(Mandatory = $true, ParameterSetName = "Path")]
    [string]$ScriptPath,

    [Parameter(Mandatory = $true, ParameterSetName = "Text")]
    [string]$ScriptText,

    [string]$Workdir = "~/project/se3_wheel_leg",

    [switch]$NoWorkdir,

    [switch]$UseProxy,

    [string]$RemoteProxy = "http://127.0.0.1:17890",

    [string]$KubePod = "",

    [string]$KubeNamespace = "",

    [string]$KubeContainer = "",

    [string[]]$SshArgs = @(),

    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function ConvertTo-BashSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function ConvertTo-BashPathExpression {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    if ($Value -eq "~") {
        return '"$HOME"'
    }
    if ($Value.StartsWith("~/")) {
        $rest = $Value.Substring(2).Replace('"', '\"')
        return '"$HOME/' + $rest + '"'
    }
    return ConvertTo-BashSingleQuoted $Value
}

function Get-ScriptBody {
    if ($PSCmdlet.ParameterSetName -eq "Path") {
        $fullPath = [System.IO.Path]::GetFullPath($ScriptPath)
        if (-not [System.IO.File]::Exists($fullPath)) {
            throw "脚本文件不存在: $fullPath"
        }
        return [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8)
    }
    return $ScriptText
}

$body = New-Object System.Collections.Generic.List[string]
$body.Add("set -euo pipefail")

if ($UseProxy) {
    $quotedProxy = ConvertTo-BashSingleQuoted $RemoteProxy
    $body.Add("export HTTP_PROXY=$quotedProxy HTTPS_PROXY=$quotedProxy")
    $body.Add("export http_proxy=$quotedProxy https_proxy=$quotedProxy")
}

if (-not $NoWorkdir -and -not [string]::IsNullOrWhiteSpace($Workdir)) {
    $body.Add("cd $(ConvertTo-BashPathExpression $Workdir)")
}

$body.Add((Get-ScriptBody))
$payload = ($body -join "`n") + "`n"
$payloadBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$encodedPayload = [System.Convert]::ToBase64String($payloadBytes)

$bootstrap = New-Object System.Collections.Generic.List[string]
$bootstrap.Add("set -euo pipefail")
$bootstrap.Add("payload='${encodedPayload}'")

if (-not [string]::IsNullOrWhiteSpace($KubePod)) {
    $kubectl = "kubectl"
    if (-not [string]::IsNullOrWhiteSpace($KubeNamespace)) {
        $kubectl += " -n " + (ConvertTo-BashSingleQuoted $KubeNamespace)
    }
    $kubectl += " exec -i " + (ConvertTo-BashSingleQuoted $KubePod)
    if (-not [string]::IsNullOrWhiteSpace($KubeContainer)) {
        $kubectl += " -c " + (ConvertTo-BashSingleQuoted $KubeContainer)
    }
    $kubectl += " -- bash -s < " + '"$tmp"'

    $bootstrap.Add("tmp=`$(mktemp)")
    $bootstrap.Add('trap ''rm -f "$tmp"'' EXIT')
    $bootstrap.Add('printf ''%s'' "$payload" | base64 -d > "$tmp"')
    $bootstrap.Add($kubectl)
} else {
    $bootstrap.Add('printf ''%s'' "$payload" | base64 -d | bash -s')
}

$bootstrapScript = ($bootstrap -join "`n") + "`n"

if ($DryRun) {
    Write-Host "host=$HostAlias"
    if ([string]::IsNullOrWhiteSpace($KubePod)) {
        Write-Host "mode=remote-bash"
    } else {
        Write-Host "mode=kubectl-exec"
        Write-Host "pod=$KubePod"
    }
    Write-Host "payload_bytes=$($payloadBytes.Length)"
    if ($NoWorkdir) {
        Write-Host "workdir=<disabled>"
    } else {
        Write-Host "workdir=$Workdir"
    }
    Write-Host "proxy=$([bool]$UseProxy)"
    exit 0
}

$bootstrapScript | & ssh @SshArgs $HostAlias "bash" "-s"
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}
