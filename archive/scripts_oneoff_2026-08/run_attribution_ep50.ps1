$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Gary\Desktop\jepa'
$env:PYTHONPATH = 'C:\Users\Gary\Desktop\jepa'
$PY = 'D:\jepa_phase0\.venv\Scripts\python.exe'
$HF   = 'D:\jepa_phase0\checkpoints_hf'
$ENVP = 'D:\jepa_phase0\runs\patch_mirage_envelope'
$BLOB = 'D:\jepa_phase0\runs\anatomy_v2_ep25'
$HEADS = 'D:\jepa_phase0\reports\refit_heads'

$jobs = @(
  @{tag='random_ep50';   ck="$HF\random-posfix-100ep\jepa_patch-ep050.pth.tar"},
  @{tag='oracle_ep50';   ck="$HF\oracle-anatomical-100ep\jepa_patch_oracle-ep050.pth.tar"},
  @{tag='envelope_ep50'; ck="$ENVP\jepa_patch_mirage-ep50.pth.tar"},
  @{tag='blob_ep50';     ck="$BLOB\jepa_patch_mirage-ep50.pth.tar"}
)
foreach ($j in $jobs) {
  Write-Output "`n########## $($j.tag) ##########"
  & $PY scripts\patch_attribution.py --ckpt $j.ck --head "$HEADS\$($j.tag)_head.pt" `
        --tag $j.tag --limit_test 1000 --slices_used 25 --amp --num_workers 3
}
Write-Output "`nALL ATTRIBUTION DONE"
