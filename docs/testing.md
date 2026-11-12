# Testing and repository maintenance

The active regression suite is `tests/`, selected by `pytest.ini`. It covers
training and masking contracts, evaluation identity, dataset/cache behavior,
numeric evidence checks, and manuscript release safeguards. Use a separate
environment and checkout; do not install test dependencies into a running
training environment or edit a campaign's pinned source.

## CPU validation

The GitHub Actions job uses Windows and Python 3.11 because the release tooling
also tests Windows file-sharing safeguards. In PowerShell, from the repository
root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m pytest tests -q
```

Keep pytest's default temporary directory. On Windows, nesting `--basetemp`
inside a deep checkout can exceed the release tooling's file-path limit when
it creates transaction recovery files.

The requirements file pins direct test dependencies; it is not a complete
transitive lock or a production training environment. Torch and torchvision
are installed separately from the official CPU index. The existing
`requirements-phase0*.txt` files belong to the archived semantic-teacher
sidecar and are not the active test environment.

Use a full Git clone: historical-reference tests inspect earlier commits.
Training/evaluation tests use synthetic inputs and temporary directories;
release tests also inspect checked-in evidence. Tests requiring the
private MIRAGE checkout, local evidence, or the project Tectonic compiler may
skip when those dependencies are unavailable; inspect the pytest summary.
Passing this suite does not establish GPU/AMP behavior, reproduce a training
run, validate new AUC claims, or authorize a manuscript release. It does not
run a model evaluation on the held-out Test split.

## What to preserve

`src/` contains the shared implementation, `scripts/` contains experiment and
maintenance entry points, and `configs/` records recipes. `archive/` contains
parked implementations and historical experiments. `autopilot/`, `results/`,
and paper release receipts include evidence needed to reproduce or qualify
published claims. Keep those records unless their provenance and replacement
have been explicitly checked. Generated caches and temporary test outputs
should remain untracked.

The local `full_blocks_v2` campaign runs from its own immutable source snapshot.
Repository cleanup does not retrofit that implementation onto older checkpoints
or alter a live campaign. Integrating a newer recipe requires a separate,
explicitly validated change.
