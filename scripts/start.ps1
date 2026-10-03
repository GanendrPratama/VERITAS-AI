# Starts the orchestrator API and the operator dashboard, detached from this
# shell -- Start-Process spawns a separate process that keeps running after
# this PowerShell window closes (the bash script's nohup+disown equivalent).
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path  # Start-Process ignores PowerShell's cwd for relative paths -- pin it explicitly

$venv = if ($env:VENV) { $env:VENV } else { "venv" }
# Start-Process -FilePath does not reliably resolve relative paths against the
# shell's cwd, so build absolute paths off $root instead.
$python = Join-Path $root "$venv\Scripts\python.exe"
$streamlit = Join-Path $root "$venv\Scripts\streamlit.exe"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python -- create it first (python -m venv $venv) or set `$env:VENV"
}
if (-not (Test-Path $streamlit)) {
    throw "streamlit not found at $streamlit -- install requirements first (pip install -r requirements.txt)"
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Start-Process -FilePath $python -ArgumentList @("-u", "orchestrator.py") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "logs\orchestrator.log" `
    -RedirectStandardError "logs\orchestrator.err.log"

Start-Process -FilePath $streamlit -ArgumentList @("run", "ui/app.py", "--server.port", "8501", "--server.headless", "true") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "logs\dashboard.log" `
    -RedirectStandardError "logs\dashboard.err.log"

function Wait-ForPort($name, $port, $errLog, $timeoutSec = 30) {
    Write-Host -NoNewline "waiting for $name on port $port "
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue) {
            Write-Host " up"
            return
        }
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1
    }
    Write-Host " timed out"
    Write-Host "$name didn't come up in ${timeoutSec}s -- check $errLog"
}

Wait-ForPort "orchestrator" 8000 "logs\orchestrator.err.log"
Wait-ForPort "dashboard" 8501 "logs\dashboard.err.log"

Write-Host "orchestrator: http://localhost:8000  (log: logs\orchestrator.log)"
Write-Host "dashboard:    http://localhost:8501  (log: logs\dashboard.log)"
Write-Host "stop with:    see README.md"
