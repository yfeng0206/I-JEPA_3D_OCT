# Extract all four ep50 arms CONCURRENTLY.
#
# The encoder pass is forward-only (~1.5-2 GB VRAM each), so a 24 GB card runs
# four at once comfortably; the sequential version left the GPU mostly idle
# waiting on volume I/O.  Each arm writes to its OWN output dir because
# downstream_region_auc.py rewrites region_auc.json wholesale and four processes
# sharing it would race.  Results are merged afterwards by merge_region_auc.py.
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Gary\Desktop\jepa'
$env:PYTHONPATH = 'C:\Users\Gary\Desktop\jepa'
$PY = 'D:\jepa_phase0\.venv\Scripts\python.exe'

$HF   = 'D:\jepa_phase0\checkpoints_hf'
$ENVP = 'D:\jepa_phase0\runs\patch_mirage_envelope'
$BLOB = 'D:\jepa_phase0\runs\anatomy_v2_ep25'
$OUT  = 'D:\jepa_phase0\reports\downstream_region_auc'

$jobs = @(
  @{tag='random_ep50';   ck="$HF\random-posfix-100ep\jepa_patch-ep050.pth.tar"},
  @{tag='oracle_ep50';   ck="$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep050.pth.tar"},
  @{tag='envelope_ep50'; ck="$ENVP\jepa_patch_mirage-ep50.pth.tar"},
  @{tag='blob_ep50';     ck="$BLOB\jepa_patch_mirage-ep50.pth.tar"}
)

$procs = @()
$MAXC = 2   # 4-way drove free RAM to 2 GB; the GPU was already pinned at 100%,
            # so concurrency beyond 2 bought throughput we could not spend and
            # risked swapping.  2 leaves ~9 GB headroom.
foreach ($j in $jobs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $MAXC) {
    Start-Sleep -Seconds 20
  }
  $o = Join-Path $OUT $j.tag
  New-Item -ItemType Directory -Force -Path $o | Out-Null
  $log = Join-Path $o 'run.log'
  $a = @('scripts\downstream_region_auc.py',
         '--ckpts', $j.ck, '--tags', $j.tag,
         '--out', $o, '--chunk', '100', '--num_workers', '2', '--amp',
         '--limit_train', '2000', '--limit_val', '600', '--limit_test', '1000',
         '--slices_used', '25')
  Write-Output "launching $($j.tag) at $(Get-Date -Format HH:mm:ss)"
  $procs += Start-Process -FilePath $PY -ArgumentList $a -PassThru `
              -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
              -WindowStyle Hidden
  Start-Sleep -Seconds 25   # stagger so they don't all hit the loader at once
}

Write-Output "PIDs: $($procs.Id -join ', ')"
foreach ($p in $procs) { $p.WaitForExit() }
Write-Output "all extraction processes exited"

& $PY scripts\merge_region_auc.py
