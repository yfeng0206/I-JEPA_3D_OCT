# Mask cache (CPU) then the four ep50 arms (GPU, one process, serialised).
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Gary\Desktop\jepa'
$env:PYTHONPATH = 'C:\Users\Gary\Desktop\jepa'
$PY = 'D:\jepa_phase0\.venv\Scripts\python.exe'

$HF   = 'D:\jepa_phase0\checkpoints_hf'
$ENVP = 'D:\jepa_phase0\runs\patch_mirage_envelope'
$BLOB = 'D:\jepa_phase0\runs\anatomy_v2_ep25'

Write-Output "########## STEP 1: anatomy mask cache ##########"
& $PY scripts\downstream_region_auc.py --build_masks
if ($LASTEXITCODE -ne 0) { throw "mask cache build failed" }

Write-Output "`n########## STEP 2: ep50 arms ##########"
& $PY scripts\downstream_region_auc.py `
  --ckpts "$HF\random-posfix-100ep\jepa_patch-ep050.pth.tar" `
          "$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep050.pth.tar" `
          "$ENVP\jepa_patch_mirage-ep50.pth.tar" `
          "$BLOB\jepa_patch_mirage-ep50.pth.tar" `
  --tags random_ep50 oracle_ep50 envelope_ep50 blob_ep50
if ($LASTEXITCODE -ne 0) { throw "ep50 region AUC failed" }
Write-Output "`nDONE"
