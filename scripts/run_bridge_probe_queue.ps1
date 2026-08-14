$ErrorActionPreference = 'Continue'
$py   = 'D:\jepa_phase0\.venv\Scripts\python.exe'
$repo = 'C:\Users\Gary\Desktop\jepa'
$env:PYTHONPATH = $repo

# Wait for the already-running ep40 probe to finish
while (Get-Process -Id 12928 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }

foreach ($tag in 'ep35','ep41') {
    $cfg = "configs\frozen_meanpool_bridge_$tag.yaml"
    $log = "D:\jepa_phase0\runs\probe_bridge_$tag.log"
    "=== starting $tag at $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append 'D:\jepa_phase0\runs\probe_queue.log'
    $p = Start-Process -FilePath $py `
        -ArgumentList '-u', 'src\eval_downstream.py', '--config', $cfg `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden
    $p.WaitForExit()
    "=== finished $tag at $(Get-Date -Format 'HH:mm:ss') exit=$($p.ExitCode) ===" | Out-File -Append 'D:\jepa_phase0\runs\probe_queue.log'
}
"=== QUEUE COMPLETE $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append 'D:\jepa_phase0\runs\probe_queue.log'
