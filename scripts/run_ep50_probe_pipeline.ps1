$ErrorActionPreference = 'Continue'
$py     = 'D:\jepa_phase0\.venv\Scripts\python.exe'
$repo   = 'C:\Users\Gary\Desktop\jepa'
$rundir = 'D:\jepa_phase0\runs\anatomy_v2_ep25'
$status = 'D:\jepa_phase0\runs\ep50_pipeline.log'
$env:PYTHONPATH = $repo
$t0 = Get-Date

function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Out-File -Append $status -Encoding ascii }

function Stop-Tree($rootPid) {
    $all = @(); $frontier = @($rootPid)
    while ($frontier.Count -gt 0) {
        $next = @()
        foreach ($p in $frontier) {
            $all += $p
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$p" -ErrorAction SilentlyContinue |
                ForEach-Object { $next += $_.ProcessId }
        }
        $frontier = $next
    }
    [array]::Reverse($all)
    foreach ($p in $all) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
}

function Start-Training {
    $ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "$rundir\train_bridge_$ts.log"
    $p = Start-Process -FilePath $py `
        -ArgumentList '-u', 'src\train_patch.py', '--config', 'configs\patch_anatomy_v2.yaml' `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden
    $log | Out-File "$rundir\CURRENT_LOG.txt" -Encoding ascii
    $p.Id | Out-File "$rundir\CURRENT_PID.txt" -Encoding ascii
    Log "TRAINING started PID $($p.Id)  log=$log"
    return $p.Id
}

Log "=== ep50 pipeline start ==="
$ck50 = "$rundir\jepa_patch_mirage-ep50.pth.tar"
$tpid = Start-Training
Log "waiting for ep50 archive ..."

while ($true) {
    Start-Sleep -Seconds 60
    if ((Test-Path $ck50) -and ((Get-Item $ck50).LastWriteTime -gt $t0)) {
        $s1 = (Get-Item $ck50).Length
        Start-Sleep -Seconds 45
        $s2 = (Get-Item $ck50).Length
        if ($s1 -eq $s2 -and $s2 -gt 1GB) {
            Log ("ep50 archive complete ({0:N2} GB)" -f ($s2 / 1GB))
            break
        }
    }
    if (-not (Get-Process -Id $tpid -ErrorAction SilentlyContinue)) {
        Log "WATCHDOG: training process gone - relaunching"
        $tpid = Start-Training
    }
}

Log "stopping training tree (root $tpid)"
Stop-Tree $tpid
Start-Sleep -Seconds 25
Log "training stopped; GPU should be free"

$plog = 'D:\jepa_phase0\runs\probe_bridge_ep50.log'
Log "PROBE ep50 starting"
$pp = Start-Process -FilePath $py `
    -ArgumentList '-u', 'src\eval_downstream.py', '--config', 'configs\frozen_meanpool_bridge_ep50.yaml' `
    -RedirectStandardOutput $plog -RedirectStandardError "$plog.err" `
    -WorkingDirectory $repo -PassThru -WindowStyle Hidden
$pp.WaitForExit()
Log "PROBE ep50 finished exit=$($pp.ExitCode)"

$res = 'D:\jepa_phase0\runs\frozen_meanpool_bridge_ep50\results.json'
if (Test-Path $res) {
    $j = Get-Content $res -Raw | ConvertFrom-Json
    Log ("RESULT ep50  test_auc={0:N4}  val_auc={1:N4}  (envelope ep50 bar = 0.8761)" -f $j.test_auc, $j.best_val_auc)
} else {
    Log "RESULT MISSING - check $plog.err"
}

Start-Sleep -Seconds 15
$tpid2 = Start-Training
Log "RESUMED training after probe (PID $tpid2), continuing to ep100"
Log "=== pipeline complete ==="
