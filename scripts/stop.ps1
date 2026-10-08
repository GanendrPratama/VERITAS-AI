# Stops the orchestrator and dashboard started by start.ps1 (whole process
# trees, force-killed), then verifies nothing is left running or listening
# on 8000/8501.
$ports = 8000, 8501

function Get-Targets {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and
        ($_.CommandLine -match 'supervise\.py' -or $_.CommandLine -match 'orchestrator\.py' -or $_.CommandLine -match 'streamlit.*run.*ui[\\/]app\.py')
    }
}
function Get-PortOwners {
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $ports -contains $_.LocalPort } | Select-Object -ExpandProperty OwningProcess -Unique
}

for ($i = 0; $i -lt 3; $i++) {
    # supervisor first, or it would restart the orchestrator we're about to kill
    foreach ($t in @(Get-Targets | Where-Object { $_.CommandLine -match 'supervise\.py' })) { taskkill /PID $t.ProcessId /T /F 2>&1 | Out-Null }
    $ids = @(Get-Targets | ForEach-Object { $_.ProcessId }) + @(Get-PortOwners) | Sort-Object -Unique
    if (-not $ids) { break }
    foreach ($id in $ids) { taskkill /PID $id /T /F 2>&1 | Out-Null }  # /T also kills child processes
    Start-Sleep -Seconds 1
}

$left = @(Get-Targets) + @(Get-PortOwners)
if ($left.Count) {
    Write-Host "FAILED: processes or ports still in use"
    Get-Targets | Format-Table ProcessId, CommandLine -AutoSize
    Get-PortOwners | ForEach-Object { Write-Host "port owner pid $_" }
    exit 1
}
Write-Host "orchestrator and dashboard stopped; ports 8000/8501 free"
