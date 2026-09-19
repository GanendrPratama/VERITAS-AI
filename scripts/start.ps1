# Starts the orchestrator API and the operator dashboard, detached from this
# shell -- Start-Process spawns a separate process that keeps running after
# this PowerShell window closes (the bash script's nohup+disown equivalent).
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path  # Start-Process ignores PowerShell's cwd for relative paths -- pin it explicitly

$venv = if ($env:VENV) { $env:VENV } else { "venv" }
$python = Join-Path $venv "Scripts\python.exe"
$streamlit = Join-Path $venv "Scripts\streamlit.exe"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Start-Process -FilePath $python -ArgumentList "orchestrator.py" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "logs\orchestrator.log" `
    -RedirectStandardError "logs\orchestrator.err.log"

Start-Process -FilePath $streamlit -ArgumentList @("run", "ui/app.py", "--server.port", "8501", "--server.headless", "true") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "logs\dashboard.log" `
    -RedirectStandardError "logs\dashboard.err.log"

Write-Host "orchestrator: http://localhost:8000  (log: logs\orchestrator.log)"
Write-Host "dashboard:    http://localhost:8501  (log: logs\dashboard.log)"
Write-Host "stop with:    see README.md"
