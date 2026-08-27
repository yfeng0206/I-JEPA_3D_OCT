# Runs the background-signal probe across every reachable checkpoint.
# GPU-serialised on purpose: one python process, small batch, forward-only.
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Gary\Desktop\jepa'
$env:PYTHONPATH = 'C:\Users\Gary\Desktop\jepa'

$HF   = 'D:\jepa_phase0\checkpoints_hf'
$ENVP = 'D:\jepa_phase0\runs\patch_mirage_envelope'
$BLOB = 'D:\jepa_phase0\runs\anatomy_v2_ep25'
$FORK = 'D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25'

$pairs = @(
  @('fork_ep25',      "$FORK\jepa_patch-random_posfix-ep25.pth.tar"),
  @('random_ep50',    "$HF\random-posfix-100ep\jepa_patch-ep050.pth.tar"),
  @('random_ep75',    "$HF\random-posfix-100ep\jepa_patch-ep075.pth.tar"),
  @('random_ep100',   "$HF\random-posfix-100ep\jepa_patch-ep100.pth.tar"),
  @('oracle_ep50',    "$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep050.pth.tar"),
  @('oracle_ep75',    "$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep075.pth.tar"),
  @('oracle_ep100',   "$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep100.pth.tar"),
  @('envelope_ep30',  "$ENVP\jepa_patch_mirage-ep30.pth.tar"),
  @('envelope_ep50',  "$ENVP\jepa_patch_mirage-ep50.pth.tar"),
  @('envelope_ep75',  "$ENVP\jepa_patch_mirage-ep75.pth.tar"),
  @('envelope_ep100', "$ENVP\jepa_patch_mirage-ep100.pth.tar"),
  @('blob_ep30',      "$BLOB\jepa_patch_mirage-ep30.pth.tar"),
  @('blob_ep40',      "$BLOB\jepa_patch_mirage-ep40.pth.tar"),
  @('blob_ep50',      "$BLOB\jepa_patch_mirage-ep50.pth.tar"),
  @('blob_ep56',      "$BLOB\jepa_patch_mirage-last.pth.tar")
)

$tags = @(); $ckpts = @()
foreach ($p in $pairs) {
  if (Test-Path $p[1]) { $tags += $p[0]; $ckpts += $p[1] }
  else { Write-Output "MISSING (skipped): $($p[0]) -> $($p[1])" }
}
Write-Output "running $($tags.Count) checkpoints"

& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\background_signal_probe.py `
    --ckpts $ckpts --tags $tags `
    --volumes 12 --slices_per_volume 8 --draws 4 --batch_size 8 `
    --out 'D:\jepa_phase0\reports\background_signal'
