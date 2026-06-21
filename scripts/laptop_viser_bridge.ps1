param(
    [int]$BridgePort = 18196,
    [int]$TunnelPort = 18197,
    [string]$RemoteSshHost = "a800",
    [string]$Target = "172.16.6.130:8080"
)

$ErrorActionPreference = "Stop"
$WorkRoot = "E:\se3_stair_viewer_setup"
$TempRoot = Join-Path $WorkRoot "tmp"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

foreach ($port in @($BridgePort, $TunnelPort)) {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -eq "ssh.exe" -or $_.Name -eq "node.exe") -and
            $_.CommandLine -like "*$port*"
        } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

$nodePath = Join-Path $TempRoot "se3_viser_bridge_$BridgePort.js"
$nodeCodeTemplate = @'
const net = require("net");
const listenPort = __BRIDGE_PORT__;
const targetPort = __TUNNEL_PORT__;

const server = net.createServer((client) => {
  const upstream = net.connect({ host: "127.0.0.1", port: targetPort });
  client.setNoDelay(true);
  upstream.setNoDelay(true);
  client.pipe(upstream);
  upstream.pipe(client);
  const close = () => {
    client.destroy();
    upstream.destroy();
  };
  client.on("error", close);
  upstream.on("error", close);
});

server.on("error", (err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});

server.listen(listenPort, "127.0.0.1", () => {
  console.log(`bridge listening 127.0.0.1:${listenPort} -> 127.0.0.1:${targetPort}`);
});
'@
$nodeCode = $nodeCodeTemplate.
    Replace("__BRIDGE_PORT__", [string]$BridgePort).
    Replace("__TUNNEL_PORT__", [string]$TunnelPort)
Set-Content -LiteralPath $nodePath -Value $nodeCode -Encoding UTF8

$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$sshArgs = "-n -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 127.0.0.1:$TunnelPort`:$Target $RemoteSshHost"
$sshErr = Join-Path $TempRoot "se3_viser_bridge_ssh_$TunnelPort.err.log"
$nodeOut = Join-Path $TempRoot "se3_viser_bridge_node_$BridgePort.out.log"
$nodeErr = Join-Path $TempRoot "se3_viser_bridge_node_$BridgePort.err.log"
Remove-Item -Force -ErrorAction SilentlyContinue $sshErr, $nodeOut, $nodeErr

$sshProc = Start-Process -FilePath $ssh -ArgumentList $sshArgs -RedirectStandardError $sshErr -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 4
if ($sshProc.HasExited) {
    Get-Content -LiteralPath $sshErr -ErrorAction SilentlyContinue
    throw "A800/llm tunnel exited early."
}

$nodeProc = Start-Process -FilePath "node.exe" -ArgumentList $nodePath -RedirectStandardOutput $nodeOut -RedirectStandardError $nodeErr -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
if ($nodeProc.HasExited) {
    Get-Content -LiteralPath $nodeOut -ErrorAction SilentlyContinue
    Get-Content -LiteralPath $nodeErr -ErrorAction SilentlyContinue
    throw "Node bridge exited early."
}

Write-Host "ready bridge=127.0.0.1:$BridgePort target=$RemoteSshHost/$Target"
Get-Content -LiteralPath $nodeOut -ErrorAction SilentlyContinue

try {
    while (-not $sshProc.HasExited -and -not $nodeProc.HasExited) {
        Start-Sleep -Seconds 1
        $sshProc.Refresh()
        $nodeProc.Refresh()
    }
}
finally {
    Stop-Process -Id $sshProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $nodeProc.Id -Force -ErrorAction SilentlyContinue
}
