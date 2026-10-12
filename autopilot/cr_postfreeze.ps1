<#
.SYNOPSIS
  Post-freeze pipeline of the registered cr_seed_v1 replication: fail-closed, resumable,
  transcript-logged, with a mandatory human review checkpoint before the real paper changes.

.DESCRIPTION
  Phase 1 (default; nothing in main_submission.tex changes):
    preflight (time >= freeze 2026-10-22 08:00 PDT, read-only campaign state, probe dirs, late /
    in-progress runs) -> cr_stats inventory -> cr_stats unseal of every ready probe dir that has
    no receipt yet (never twice) -> cr_stats final -> cr_md_regression --split test ->
    make_cr_results cb-audit (RANDOM-CB trainer log) -> make_cr_results build --update-reviews ->
    build --check -> scratch paper copy -> integrate -> Tectonic compile (Idle) -> receipt
    preview (p15 + migration plan on the scratch copy) -> poster scratch build -> REVIEW.md,
    then STOP for human review.
  Phase 2 (-ApplyToPaper, after the review):
    build --check -> integrate the real paper (--allow-real-paper) -> register the seed-spread
    figure (release_assets.json producer + p15 figure receipt, -FigureReviewer) -> p15 ->
    literal receipt migration (automatic only for paragraphs changed solely by the integration
    replacements; any other changed claim context needs -AcceptReceipts <id> with a named
    -FigureReviewer) -> p15 -> p13 --camera-ready (Idle, so its Tectonic child is Idle too).
  Every native step must exit 0 (documented exceptions: the receipt preview's p15 and the
  poster report); the first failure stops the run. Re-running is safe: unseal skips receipts,
  final / MD are skipped only when their inputs and the completion evidence (stop files, seal
  and probe times) are unchanged and the output still has the sha256 recorded when it was
  computed (an edited output is recomputed); a cached or fresh final must agree with the current
  evidence and its labels, contrasts, means and RANDOM range must be what the registered rules
  give for the receipt-verified per-run AUCs (check-final); integration of the real paper is
  skipped once done (registration, receipts and p13 are repeated).

  "Late" is decided exactly as cr_stats final decides it (cr_stats.completion_eligibility): a
  run is in the primary analysis only if its trainer stop file (stop_epoch_050.json, the
  hash-recorded run provenance), its seal creation and its probe finish time all predate the
  freeze. These timestamps are written by the trainer and the probe; nothing has to be supplied
  here, but stop files must stay where the probes recorded them. A seed index whose RANDOM,
  CENTROID and ENVELOPE runs are not all on time is reported descriptively. Runs still training
  or probing at the start are "in progress": the preflight stops (exit 3) until you either wait
  for them (they are then reported as finished after the freeze) or pass -AllowIncomplete
  (they are reported as not completed, pre-registration section 7).

  -Rehearsal runs both phases on a synthetic campaign (tests\cr_postfreeze_fixtures.py) against
  a mirror repository (a copy of this working tree standing in for the real one); it refuses
  every real campaign, run, worktree, paper and repository path, requires the fixture marker and
  explicit "synthetic": true provenance, and contains every derived path (state probe dirs, stop
  files, trainer logs, feature caches, sealed entries; junctions resolved) in the fixture root
  before reading it or passing --allow-before-freeze.

.EXAMPLE
  # Thursday 2026-10-22 after 08:00 PDT, repository root:
  powershell -NoProfile -ExecutionPolicy Bypass -File autopilot\cr_postfreeze.ps1
  # after reviewing D:\jepa_phase0\autopilot_out\cr_postfreeze\REVIEW.md and the scratch PDF:
  powershell -NoProfile -ExecutionPolicy Bypass -File autopilot\cr_postfreeze.ps1 -ApplyToPaper -FigureReviewer "Name, 2026-10-22"

.EXAMPLE
  # synthetic rehearsal
  python tests\cr_postfreeze_fixtures.py <scratch>\fx --late envelope:9012
  powershell -NoProfile -File autopilot\cr_postfreeze.ps1 -Rehearsal -FixtureRoot <scratch>\fx -OutRoot <scratch> -FreshMirror
  powershell -NoProfile -File autopilot\cr_postfreeze.ps1 -Rehearsal -FixtureRoot <scratch>\fx -OutRoot <scratch> -ApplyToPaper -FigureReviewer "SYNTHETIC REHEARSAL"
#>
[CmdletBinding()]
param(
  [string]$RepoRoot = '',
  [string]$TargetRepo = '',
  [string]$CampaignJson = '',
  [string]$StateJson = 'D:\jepa_phase0\campaign\cr_seed_v1\state.json',
  [string]$RunsRoot = 'D:\jepa_phase0\runs',
  [string]$OutRoot = 'D:\jepa_phase0\autopilot_out\cr_postfreeze',
  [string]$StatsDir = 'D:\jepa_phase0\autopilot_out\p1_stats',
  [string]$MdMetadata = 'D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv',
  [string]$BudgetI7 = 'C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\camera_ready\i7_scratch\real_bench.json',
  [string]$CitationRecord = '',
  [string]$PosterDir = '',
  [string]$Python = 'D:\jepa_phase0\.venv\Scripts\python.exe',
  [string]$Tectonic = 'D:\jepa_phase0\tools\tectonic\tectonic.exe',
  [int]$NBoot = 10000,
  [double]$MinFreeCommitGiB = 4,
  [switch]$AllowIncomplete,
  [switch]$SkipCbAudit,
  [switch]$SkipReceiptPreview,
  [switch]$SkipPoster,
  [switch]$ApplyToPaper,
  [string]$FigureReviewer = '',
  [string[]]$AcceptReceipts = @(),
  [string]$ReleaseOut = '',
  [switch]$SkipRelease,
  [switch]$Rehearsal,
  [string]$FixtureRoot = '',
  [switch]$FreshMirror,
  [switch]$RehearsalFillAcknowledgments,
  [switch]$LibraryOnly
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$Freeze = [DateTimeOffset]::Parse('2026-10-22T08:00:00-07:00')
$Phase = if ($ApplyToPaper) { 'apply' } else { 'review' }

# ---------------------------------------------------------------------------- helpers
function Get-Full([string]$Path) { [IO.Path]::GetFullPath($Path).TrimEnd('\') }
function Test-Under([string]$Path, [string]$Root) {
  $p = Get-Full $Path; $r = Get-Full $Root
  return ($p -ieq $r) -or $p.StartsWith($r + '\', [StringComparison]::OrdinalIgnoreCase)
}
function Fail([string]$Message, [int]$Code = 2) {
  Write-Host "ERROR: $Message" -ForegroundColor Red
  $script:ExitCode = $Code
  throw "STOP($Code): $Message"
}
function ConvertTo-ArgString([string[]]$Arguments) {
  ($Arguments | ForEach-Object {
      if ($_ -eq '') { '""' }
      elseif ($_ -notmatch '[\s"]') { $_ }
      else { '"' + (($_ -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"' }
    }) -join ' '
}
function Invoke-Native {
  param([string]$Exe, [string[]]$Arguments, [string]$Name, [string]$WorkDir = $RepoRoot,
        [string]$Priority = 'BelowNormal', [int]$TailLines = 0)
  $log = Join-Path $LogDir ("{0}_{1}.log" -f $Phase, ($Name -replace '[^A-Za-z0-9_.-]', '_'))
  $cmdline = "$Exe " + (ConvertTo-ArgString $Arguments)
  Write-Host "COMMAND: $cmdline"
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $Exe
  $psi.Arguments = ConvertTo-ArgString $Arguments
  $psi.WorkingDirectory = $WorkDir
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  # Children of an Idle / BelowNormal process inherit its class at creation, so the parent's
  # class is set to the requested one around Start (grandchildren such as p13's Tectonic
  # compile inherit it too), then restored; the child's class is also set explicitly.
  $me = Get-Process -Id $PID
  $own = $me.PriorityClass
  try {
    try { $me.PriorityClass = $Priority } catch { }
    $p = [System.Diagnostics.Process]::Start($psi)
  } finally {
    try { $me.PriorityClass = $own } catch { }
  }
  try { $p.PriorityClass = $Priority } catch { Write-Host "  (priority not set: $($_.Exception.Message))" }
  $actual = try { $p.PriorityClass } catch { 'unknown (exited)' }
  Write-Host "PRIORITY: $actual"
  $out = $p.StandardOutput.ReadToEndAsync()
  $err = $p.StandardError.ReadToEndAsync()
  $p.WaitForExit()
  $text = $out.Result + $(if ($err.Result) { "`n[stderr]`n" + $err.Result } else { '' })
  Set-Content -Path $log -Value ("COMMAND: $cmdline`nPRIORITY: $actual`nEXIT: $($p.ExitCode)`n" + $text) -Encoding UTF8
  if ($text.Trim()) {
    if ($TailLines -gt 0) {
      $lines = $text.TrimEnd() -split "`r?`n"
      if ($lines.Count -gt $TailLines) { Write-Host "  ... ($($lines.Count - $TailLines) lines in $log)" }
      Write-Host (($lines | Select-Object -Last $TailLines) -join "`n")
    } else { Write-Host $text.TrimEnd() }
  }
  return $p.ExitCode
}
function Invoke-Py {
  param([string]$Name, [string[]]$Arguments, [int[]]$OkCodes = @(0), [string]$WorkDir = $RepoRoot,
        [string]$Priority = 'BelowNormal')
  $code = Invoke-Native -Exe $Python -Arguments $Arguments -Name $Name -WorkDir $WorkDir -Priority $Priority
  if ($OkCodes -notcontains $code) { Fail "$Name exited with $code (log: $LogDir)" $(if ($code -eq 3) { 3 } else { 2 }) }
  return $code
}
function Wait-Memory {
  for ($i = 0; $i -lt 30; $i++) {
    $free = (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB
    if ($free -ge $MinFreeCommitGiB) { Write-Host ("  free commit {0:N1} GiB" -f $free); return }
    Write-Host ("  free commit {0:N1} GiB < {1} GiB; waiting 2 minutes" -f $free, $MinFreeCommitGiB)
    Start-Sleep -Seconds 120
  }
  Fail "free commit stayed below $MinFreeCommitGiB GiB for an hour"
}
function Wait-Tectonic {
  for ($i = 0; $i -lt 60; $i++) {
    $running = @(Get-Process -Name tectonic -ErrorAction SilentlyContinue)
    if (-not $running) { return }
    Write-Host "  another Tectonic compile is running (PID $($running.Id -join ', ')); waiting 1 minute"
    Start-Sleep -Seconds 60
  }
  Fail 'another Tectonic compile kept running for an hour'
}
function Invoke-Step([string]$Name, [scriptblock]$Body) {
  Write-Host ''
  Write-Host ("=== [{0}] {1} start {2}" -f $Phase, $Name, (Get-Date -Format 's')) -ForegroundColor Cyan
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $outcome = 'ok'
  try {
    $result = & $Body
    if ($null -ne $result) { $outcome = ($result | Select-Object -Last 1).ToString() }
  } catch {
    $outcome = 'FAILED: ' + $_.Exception.Message
    throw
  } finally {
    $sw.Stop()
    $script:Steps += , ([ordered]@{ phase = $Phase; step = $Name; started = (Get-Date).AddMilliseconds(-$sw.ElapsedMilliseconds).ToString('s');
                                    seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); outcome = $outcome })
    ConvertTo-Json -InputObject @($script:Steps) -Depth 4 | Set-Content -Path $StepsFile -Encoding UTF8
    Write-Host ("=== [{0}] {1}: {2} ({3:N1} s)" -f $Phase, $Name, $outcome, $sw.Elapsed.TotalSeconds) -ForegroundColor Cyan
  }
}
function Read-Json([string]$Path) { Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json }
function Get-Prop($Object, [string]$Name) {
  $p = $Object.PSObject.Properties[$Name]
  if ($p) { return $p.Value }
  return $null
}
# V10-1: a cache record binds the input key AND the digest of the output it produced, so an
# output edited after it was computed is never reused (it is recomputed, with a warning).
function Get-Sha256([string]$Path) {
  # .NET instead of Get-FileHash: Windows PowerShell started from pwsh 7 may not autoload it.
  $stream = [IO.File]::OpenRead((Get-Full $Path))
  try { -join ([Security.Cryptography.SHA256]::Create().ComputeHash($stream) | ForEach-Object { $_.ToString('x2') }) }
  finally { $stream.Dispose() }
}
function Test-CacheValid([string]$Output, [string]$KeyFile, [string]$Key) {
  if (-not (Test-Path $Output) -or -not (Test-Path $KeyFile)) { return $false }
  $parts = @(((Get-Content -Raw $KeyFile).Trim()) -split '\s+')
  if ($parts.Count -ne 2 -or $parts[0] -ne $Key) { return $false }
  $sha = Get-Sha256 $Output
  if ($parts[1] -ne $sha) {
    Write-Host "WARNING: $Output changed after it was computed (sha256 $sha, recorded $($parts[1])); recomputing" -ForegroundColor Yellow
    return $false
  }
  return $true
}
function Save-CacheRecord([string]$Output, [string]$KeyFile, [string]$Key) {
  if (-not (Test-Path $Output)) { Fail "$Output was not written" }
  Set-Content -Path $KeyFile -Value ('{0} {1}' -f $Key, (Get-Sha256 $Output)) -Encoding ascii
}
function Get-InputKey([string[]]$Extra) {
  $key = & $Python "$Tools\cr_postfreeze.py" input-key --plan (Join-Path $OutRoot 'plan.json') --extra @Extra
  if ($LASTEXITCODE -ne 0 -or -not $key) { Fail 'input-key failed' }
  return ($key | Select-Object -Last 1).Trim()
}

function Assert-Contained([string[]]$Paths, [string]$What) {
  # Rehearsal (V8-1): every probe dir handed to a tool with --allow-before-freeze must resolve
  # inside the fixture root (string check here, junction-resolving check in cr_postfreeze.py).
  if (-not $Rehearsal) { return }
  foreach ($p in $Paths) {
    if (-not (Test-Under $p $FixtureRoot)) { Fail "$What $p is outside the fixture root $FixtureRoot" }
  }
  Invoke-Py ('contain_' + ($What -replace '[^A-Za-z0-9]', '_')) (@("$Tools\cr_postfreeze.py", 'contain', '--root', $FixtureRoot, '--paths') + $Paths) | Out-Null
}

if ($LibraryOnly) { return }

# ---------------------------------------------------------------------------- configuration
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = Get-Full $RepoRoot
if (-not $CampaignJson) { $CampaignJson = Join-Path $RepoRoot 'configs\cr_seed_v1\campaign.json' }
if (-not $CitationRecord) { $CitationRecord = 'autopilot\investigations\camera_ready_20261008\paper_i8\citation_authorities_cr.json' }
if (-not $PosterDir) { $PosterDir = Join-Path $RepoRoot 'poster\genai4health2026' }
$RealPaper = Join-Path $RepoRoot 'paper\genai4health2026'
$RealRuns = 'D:\jepa_phase0\runs'
$RealCampaign = 'D:\jepa_phase0\campaign'
$RealWorktrees = 'D:\jepa_phase0\cr_worktrees'
if ($Rehearsal) {
  if (-not $FixtureRoot) { throw 'Rehearsal needs -FixtureRoot (tests\cr_postfreeze_fixtures.py output)' }
  $FixtureRoot = Get-Full $FixtureRoot
  $CampaignJson = Join-Path $FixtureRoot 'campaign.json'
  $StateJson = Join-Path $FixtureRoot 'state.json'
  $RunsRoot = Join-Path $FixtureRoot 'runs'
  $MdMetadata = Join-Path $FixtureRoot 'metadata.csv'
  if ($OutRoot -ieq 'D:\jepa_phase0\autopilot_out\cr_postfreeze') { throw 'Rehearsal needs an explicit scratch -OutRoot' }
  if (-not $TargetRepo) { $TargetRepo = Join-Path $OutRoot 'm' }
  $FreezeArgs = @('--allow-before-freeze')
} else {
  if ($FixtureRoot) { throw '-FixtureRoot is for -Rehearsal only' }
  if (-not $TargetRepo) { $TargetRepo = $RepoRoot }
  $FreezeArgs = @()
}
$OutRoot = Get-Full $OutRoot
$TargetRepo = Get-Full $TargetRepo
$TargetPaper = Join-Path $TargetRepo 'paper\genai4health2026'
$ResultsDir = Join-Path $TargetPaper 'auto'
$EvidenceDir = Join-Path $TargetRepo 'autopilot\investigations\camera_ready_20261008\results_cr_seed_v1'
$ScratchPaper = Join-Path $OutRoot 'scratch\paper\genai4health2026'
$ScratchPoster = Join-Path $OutRoot 'scratch\poster\genai4health2026'
$Snapshot = Join-Path $OutRoot 'pre_integration_paper'
$LogDir = Join-Path $OutRoot 'logs'
$StepsFile = Join-Path $OutRoot "steps_$Phase.json"
$Tools = Join-Path $RepoRoot 'autopilot'
$GateTools = Join-Path $TargetRepo 'autopilot'
$Steps = @()
$ExitCode = 0

# Fail-closed path policy (checked before anything is written).
if ($Rehearsal) {
  foreach ($p in @($FixtureRoot, $CampaignJson, $StateJson, $RunsRoot, $MdMetadata, $OutRoot, $TargetRepo)) {
    foreach ($forbidden in @($RealRuns, $RealCampaign, $RealWorktrees, $RepoRoot)) {
      if (Test-Under $p $forbidden) { throw "Rehearsal refuses real path $p (under $forbidden)" }
    }
  }
  if ($TargetRepo -ieq $RepoRoot) { throw 'Rehearsal needs a mirror repository, not the real one' }
  # Deepest rehearsal path: <mirror>\autopilot\investigations\camera_ready_20261008\results_cr_seed_v1\receipts\<anchor probe dir>.unsealed_results.json
  if ($TargetRepo.Length + 170 -gt 259) { throw "Mirror path $TargetRepo is too long for MAX_PATH (LongPathsEnabled=0); pass a short -TargetRepo such as C:\t\cr_mirror" }
} else {
  if ($TargetRepo -ine $RepoRoot) { throw 'The real run edits this repository; -TargetRepo is for -Rehearsal' }
  if ((Test-Under $OutRoot $RepoRoot) -or (Test-Under $OutRoot $RealRuns)) { throw "-OutRoot $OutRoot must be outside the repository and the run root" }
  if ([DateTimeOffset]::Now -lt $Freeze) {
    throw "Refusing to run before the results freeze ($Freeze); use -Rehearsal for synthetic fixtures"
  }
}

New-Item -ItemType Directory -Force -Path $OutRoot, $LogDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'; $env:OPENBLAS_NUM_THREADS = '2'
$env:MPLBACKEND = 'Agg'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }
$Transcript = Join-Path $LogDir ("postfreeze_{0}_{1}.transcript.txt" -f $Phase, (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $Transcript | Out-Null
Write-Host ("cr_postfreeze phase={0} rehearsal={1} repo={2} target={3} out={4}" -f $Phase, [bool]$Rehearsal, $RepoRoot, $TargetRepo, $OutRoot)

$BuildArgsFile = Join-Path $OutRoot 'build_args.json'
function Get-BuildArgs {
  $build = @('build', '--final', (Join-Path $OutRoot 'final.json'), '--inventory', (Join-Path $OutRoot 'inventory.json'),
             '--md', (Join-Path $OutRoot 'md_test.json'), '--stats-dir', $StatsDir,
             '--out-dir', $ResultsDir, '--evidence-dir', $EvidenceDir)
  if (Test-Path (Join-Path $OutRoot 'cb_audit.json')) { $build += @('--budget-cb', (Join-Path $OutRoot 'cb_audit.json')) }
  # I5 budgets (RANDOM, CENTROID rows) are always passed: with --budget-cb alone the generator
  # would drop them (the CB audit only has the RANDOM-CB row).
  $i5 = Join-Path $RepoRoot 'autopilot\investigations\camera_ready_20261008\mask_audit\delivered_mask_audit_trained_3seeds_100draws.json'
  if (-not (Test-Path $i5)) { Fail "I5 delivered-mask audit missing: $i5" }
  $build += @('--budget-i5', $i5)
  if ($BudgetI7) {
    # Explicit -BudgetI7 '' skips the I7 bench (purity and per-image match counts then come only
    # from a RANDOM-CB audit); a configured but missing file stops the run.
    if (-not (Test-Path $BudgetI7)) { Fail "I7 matcher bench missing: $BudgetI7 (pass -BudgetI7 '' to build without it)" }
    $build += @('--budget-i7', $BudgetI7)
  }
  if ($Rehearsal) { $build += '--allow-synthetic' }
  return , $build
}
function Read-BuildArgs {
  # ConvertFrom-Json (Windows PowerShell 5.1) emits a JSON array as one object; unroll it.
  $j = Read-Json $BuildArgsFile
  return , @($j | ForEach-Object { [string]$_ })
}
function Test-Integrated { (Get-Content -Raw -Encoding UTF8 (Join-Path $TargetPaper 'main_submission.tex')).Contains('\input{auto/cr_results}') }

try {
  if (-not $ApplyToPaper) {
    # ======================================================================== phase 1
    Invoke-Step 'preflight' {
      Wait-Memory
      # H8: the registered MD secondary analysis always runs (md-test, then build --md); its
      # metadata must exist before anything is unsealed.
      if (-not $Rehearsal -and -not (Test-Path -LiteralPath $MdMetadata)) { Fail "MD metadata missing: $MdMetadata (needed by the registered MD secondary analysis)" }
      # The plan comes first: in a rehearsal it contains every derived path in the fixture root
      # (junctions resolved) before anything else is read or copied (V8-1).
      $planArgs = @("$Tools\cr_postfreeze.py", 'plan', '--campaign', $CampaignJson, '--state', $StateJson,
                    '--runs-root', $RunsRoot, '--out', (Join-Path $OutRoot 'plan.json'))
      if ($AllowIncomplete) { $planArgs += '--allow-in-progress' }
      if ($Rehearsal) { $planArgs += @('--rehearsal', '--fixture-root', $FixtureRoot, '--inputs', $MdMetadata) }
      Invoke-Py 'plan' $planArgs | Out-Null
      $plan = Read-Json (Join-Path $OutRoot 'plan.json')
      Assert-Contained @($plan.ready_dirs) 'ready probe dirs'
      if ($Rehearsal -and ($FreshMirror -or -not (Test-Path (Join-Path $TargetRepo 'CR_POSTFREEZE_REHEARSAL_MIRROR.json')))) {
        Invoke-Py 'rehearsal-mirror' @("$Tools\cr_postfreeze.py", 'rehearsal-mirror', '--repo', $RepoRoot, '--dest', $TargetRepo) | Out-Null
      }
      if ($Rehearsal -and -not (Test-Path (Join-Path $TargetRepo 'CR_POSTFREEZE_REHEARSAL_MIRROR.json'))) { Fail 'mirror marker missing' }
      if (Test-Integrated) {
        Fail "$TargetPaper is already integrated. Run -ApplyToPaper, or restore main_submission.tex and compact_protocol.tex (git checkout) before re-running phase 1."
      }
      "ready $(@($plan.ready_dirs).Count), to unseal $(@($plan.to_unseal).Count), late [$(@($plan.late) -join ',')], in progress [$(@($plan.in_progress) -join ',')], not completed [$(@($plan.not_completed) -join ',')]"
    }
    $plan = Read-Json (Join-Path $OutRoot 'plan.json')
    $ready = @($plan.ready_dirs)

    Invoke-Step 'inventory' {
      Invoke-Py 'inventory' (@("$Tools\cr_stats.py", 'inventory', '--runs') + $ready + @('--out', (Join-Path $OutRoot 'inventory.json'))) | Out-Null
      "$($ready.Count) probe dirs"
    }

    Invoke-Step 'unseal' {
      $opened = 0; $skipped = 0
      foreach ($dir in $ready) {
        if (Test-Path (Join-Path $dir 'unsealed_results.json')) {
          Write-Host "  already unsealed, not opened again: $dir"; $skipped++; continue
        }
        Assert-Contained @($dir) 'probe dir to unseal'  # never --allow-before-freeze outside the fixture
        Invoke-Py ('unseal_' + (Split-Path $dir -Leaf)) (@("$Tools\cr_stats.py", 'unseal', '--run', $dir) + $FreezeArgs) | Out-Null
        if (-not (Test-Path (Join-Path $dir 'unsealed_results.json'))) { Fail "no receipt after unsealing $dir" }
        $opened++
      }
      "unsealed $opened, already unsealed $skipped"
    }

    Invoke-Step 'final' {
      # The key covers the completion evidence too (stop files, seal / probe times; V8-2).
      $key = Get-InputKey @("$Tools\cr_stats.py", "n_boot=$NBoot")
      $keyFile = Join-Path $OutRoot 'final.key'
      $outcome = 'computed'
      if (Test-CacheValid (Join-Path $OutRoot 'final.json') $keyFile $key) {
        $outcome = 'skipped (inputs and output unchanged)'
      } else {
        Wait-Memory
        Assert-Contained $ready 'probe dirs for final'
        Invoke-Py 'final' (@("$Tools\cr_stats.py", 'final', '--inventory', (Join-Path $OutRoot 'inventory.json'),
                            '--out', (Join-Path $OutRoot 'final.json'), '--n-boot', "$NBoot") + $FreezeArgs) | Out-Null
        Save-CacheRecord (Join-Path $OutRoot 'final.json') $keyFile $key
      }
      # Cached or fresh, final must agree with the plan's current completion evidence and with
      # the registered rules applied to the receipt-verified per-run AUCs (labels, ranges).
      Invoke-Py 'check_final' @("$Tools\cr_postfreeze.py", 'check-final', '--plan', (Join-Path $OutRoot 'plan.json'),
                                '--final', (Join-Path $OutRoot 'final.json')) | Out-Null
      $outcome
    }

    Invoke-Step 'md-test' {
      $key = Get-InputKey @("$Tools\cr_md_regression.py", $MdMetadata, "n_boot=$NBoot")
      $keyFile = Join-Path $OutRoot 'md_test.key'
      if (Test-CacheValid (Join-Path $OutRoot 'md_test.json') $keyFile $key) {
        return 'skipped (inputs and output unchanged)'
      }
      Wait-Memory
      Assert-Contained $ready 'probe dirs for MD'
      Invoke-Py 'md_test' (@("$Tools\cr_md_regression.py", '--runs') + $ready + @('--split', 'test', '--metadata', $MdMetadata,
                            '--n-boot', "$NBoot", '--out', (Join-Path $OutRoot 'md_test.json')) + $FreezeArgs) | Out-Null
      Save-CacheRecord (Join-Path $OutRoot 'md_test.json') $keyFile $key
      'computed'
    }

    Invoke-Step 'cb-audit' {
      $cbAudit = Join-Path $OutRoot 'cb_audit.json'
      if (Test-Path $cbAudit) { Remove-Item $cbAudit }
      $logs = @($plan.cb.train_logs)
      if ($SkipCbAudit) { return 'skipped (-SkipCbAudit; I5 budgets used)' }
      if ($plan.cb.status -ne 'ready' -or -not $logs) { return "skipped (RANDOM-CB status $($plan.cb.status); I5 budgets used)" }
      Invoke-Py 'cb_audit' (@("$Tools\make_cr_results.py", 'cb-audit', '--log') + $logs + @('--out', $cbAudit)) | Out-Null
      "audit from $($logs.Count) log(s)"
    }

    Invoke-Step 'build' {
      Wait-Memory
      $build = Get-BuildArgs
      ConvertTo-Json -InputObject $build | Set-Content -Path $BuildArgsFile -Encoding UTF8
      if ($Rehearsal) {
        Invoke-Py 'build' (@("$Tools\make_cr_results.py") + $build) | Out-Null
        Invoke-Py 'rehearsal_bind' @("$Tools\cr_postfreeze.py", 'rehearsal-bind', '--sidecar', (Join-Path $ResultsDir 'cr_results.values.json'),
                                     '--review-file', (Join-Path $TargetPaper 'numeric_reviews.json'), '--repo-root', $TargetRepo) | Out-Null
      } else {
        Invoke-Py 'build' (@("$Tools\make_cr_results.py") + $build + @('--update-reviews')) | Out-Null
      }
      Set-Content -Path (Join-Path $OutRoot 'results_dir.txt') -Value $ResultsDir -Encoding UTF8
      $s = (Read-Json (Join-Path $ResultsDir 'cr_results.values.json')).scenario
      "scenario $($s.scenario), matched $($s.matched_control.label)"
    }

    Invoke-Step 'build-check' {
      $build = Read-BuildArgs
      Invoke-Py 'build_check' (@("$Tools\make_cr_results.py") + $build + @('--check')) | Out-Null
    }

    Invoke-Step 'scratch-integrate' {
      Invoke-Py 'scratch_copy' @("$Tools\cr_postfreeze.py", 'scratch-copy', '--src', $TargetPaper, '--dst', $ScratchPaper) | Out-Null
      $pre = Join-Path $OutRoot 'scratch\pre_integration'
      New-Item -ItemType Directory -Force -Path $pre | Out-Null
      foreach ($f in 'main_submission.tex', 'compact_protocol.tex', 'compact_precision.tex') {
        if (Test-Path (Join-Path $ScratchPaper $f)) { Copy-Item (Join-Path $ScratchPaper $f) $pre -Force }
      }
      Invoke-Py 'integrate_scratch' @("$Tools\make_cr_results.py", 'integrate', '--paper-dir', $ScratchPaper, '--results-dir', $ResultsDir) | Out-Null
    }

    Invoke-Step 'compile-scratch' {
      Wait-Memory
      Wait-Tectonic
      foreach ($f in 'main_submission.pdf', 'main_submission.log') { if (Test-Path (Join-Path $ScratchPaper $f)) { Remove-Item (Join-Path $ScratchPaper $f) } }
      $code = Invoke-Native -Exe $Tectonic -Arguments @('-X', 'compile', 'main_submission.tex', '--keep-logs') -Name 'tectonic_scratch' -WorkDir $ScratchPaper -Priority 'Idle' -TailLines 3
      if ($code -ne 0) { Fail "Tectonic failed on the scratch copy (exit $code)" }
      Invoke-Py 'compile_report' @("$Tools\cr_postfreeze.py", 'compile-report', '--paper-dir', $ScratchPaper, '--out', (Join-Path $OutRoot 'compile_scratch.json')) | Out-Null
      $r = Read-Json (Join-Path $OutRoot 'compile_scratch.json')
      "main pages $($r.main_content_pages) (ends $($r.main_text_fraction)), undefined refs $(@($r.undefined_refs).Count), overfull $($r.overfull_boxes)"
    }

    if (-not $SkipReceiptPreview) {
      Invoke-Step 'receipts-preview' {
        Wait-Memory
        $report = Join-Path $OutRoot 'p15_scratch.json'
        if (Test-Path $report) { Remove-Item $report }
        $code = Invoke-Native -Exe $Python -Arguments @("$GateTools\p15_verify_numbers.py", '--paper-dir', $ScratchPaper, '--stats-dir', $StatsDir,
                                                         '--review-file', (Join-Path $ScratchPaper 'numeric_reviews.json'), '--report', $report) -Name 'p15_scratch' -TailLines 4
        if (-not (Test-Path $report)) { Fail 'p15 wrote no report on the scratch copy' }
        Invoke-Py 'migrate_preview' @("$Tools\cr_migrate_receipts.py", '--paper-dir', $ScratchPaper, '--old-dir', (Join-Path $OutRoot 'scratch\pre_integration'),
                                      '--results-dir', $ResultsDir, '--report', $report, '--out', (Join-Path $OutRoot 'receipts_preview.json')) | Out-Null
        $p = Read-Json (Join-Path $OutRoot 'receipts_preview.json')
        "p15 exit $code (expected non-zero before registration and migration); stale $($p.n_stale), migratable $(@($p.migrations).Count), manual $(@($p.manual).Count)"
      }
    }

    if (-not $SkipPoster) {
      Invoke-Step 'poster-scratch' {
        Invoke-Py 'poster_copy' @("$Tools\cr_postfreeze.py", 'scratch-copy', '--kind', 'poster', '--src', $PosterDir, '--dst', $ScratchPoster) | Out-Null
        Wait-Memory
        Wait-Tectonic
        $env:TECTONIC = $Tectonic
        $oldPath = $env:PATH
        $env:PATH = (Split-Path -Parent $Python) + ';' + $env:PATH
        try {
          $posterArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $ScratchPoster 'build.ps1'))
          # The poster build refuses synthetic results unless -AllowSynthetic (scratch copies only).
          if ($Rehearsal -and ((Get-Content -Raw (Join-Path $ScratchPoster 'build.ps1')) -match '\[switch\]\$AllowSynthetic')) { $posterArgs += '-AllowSynthetic' }
          # The poster README builds with pwsh; Windows PowerShell 5.1 is the fallback.
          $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { (Get-Command pwsh).Source } else { 'powershell.exe' }
          $code = Invoke-Native -Exe $shell -Arguments $posterArgs -Name 'poster_build' -WorkDir $ScratchPoster -TailLines 25
        } finally { $env:PATH = $oldPath }
        $plog = Join-Path $ScratchPoster 'poster.log'
        $loaded = (Test-Path $plog) -and ((Get-Content -Raw $plog) -match 'cr_results.tex found and loaded')
        $note = "cr_results.tex loaded: $loaded; poster.pdf: $(Test-Path (Join-Path $ScratchPoster 'poster.pdf')); see the poster build report (Placeholders / Results lines) in the poster_build log for remaining manual items"
        @{ exit_code = $code; cr_results_loaded = $loaded; note = $note } | ConvertTo-Json | Set-Content -Path (Join-Path $OutRoot 'poster_report.json') -Encoding UTF8
        if (-not (Test-Path (Join-Path $ScratchPoster 'poster.pdf'))) { Fail 'poster scratch build produced no PDF' }
        "build exit $code; cr_results loaded $loaded"
      }
    }

    Invoke-Step 'summary' { Invoke-Py 'summary' @("$Tools\cr_postfreeze.py", 'summary', '--out-root', $OutRoot) | Out-Null }
    Write-Host ''
    Write-Host 'STOP: human review required before the real paper changes.' -ForegroundColor Yellow
    Write-Host "  Review $OutRoot\REVIEW.md, the scratch PDF $ScratchPaper\main_submission.pdf (abstract, results, Table 2, control table, figure, appendix) and the display warnings."
    $next = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -ApplyToPaper -FigureReviewer `"<name, date>`""
    if ($Rehearsal) { $next += " -Rehearsal -FixtureRoot `"$FixtureRoot`" -OutRoot `"$OutRoot`" -TargetRepo `"$TargetRepo`"" }
    Write-Host "  Then: $next"
  } else {
    # ======================================================================== phase 2
    foreach ($f in 'plan.json', 'final.json', 'inventory.json', 'build_args.json', 'results_dir.txt') {
      if (-not (Test-Path (Join-Path $OutRoot $f))) { Fail "phase 1 output $f missing in $OutRoot; run phase 1 first" }
    }
    $sidecar = Read-Json (Join-Path $ResultsDir 'cr_results.values.json')
    $hasFigure = $null -ne $sidecar.figure
    if ($hasFigure -and -not $FigureReviewer.Trim()) { Fail '-FigureReviewer "<name, date>" is required: it signs the figure receipt after you inspected the rendered figure' }
    $realArg = @(); if (-not $Rehearsal) { $realArg = @('--allow-real-paper') }

    Invoke-Step 'build-check' {
      Wait-Memory
      $build = Read-BuildArgs
      Invoke-Py 'build_check' (@("$Tools\make_cr_results.py") + $build + @('--check')) | Out-Null
    }

    Invoke-Step 'integrate-paper' {
      if (Test-Integrated) {
        if (-not (Test-Path (Join-Path $Snapshot 'main_submission.tex'))) { Fail "paper already integrated but no pre-integration snapshot in $Snapshot" }
        return 'skipped (already integrated)'
      }
      New-Item -ItemType Directory -Force -Path $Snapshot | Out-Null
      foreach ($f in 'main_submission.tex', 'compact_protocol.tex', 'compact_precision.tex') {
        if (Test-Path (Join-Path $TargetPaper $f)) { Copy-Item (Join-Path $TargetPaper $f) $Snapshot -Force }
      }
      Invoke-Py 'integrate_paper' (@("$Tools\make_cr_results.py", 'integrate', '--paper-dir', $TargetPaper, '--results-dir', $ResultsDir) + $realArg) | Out-Null
      'integrated'
    }

    if ($hasFigure) {
      Invoke-Step 'register-figure' {
        Invoke-Py 'register_figure' (@("$Tools\make_cr_results.py", 'register-figure', '--paper-dir', $TargetPaper, '--results-dir', $ResultsDir,
                                       '--reviewer', $FigureReviewer, '--registry', (Join-Path $TargetRepo 'autopilot\release_assets.json'),
                                       '--review-file', (Join-Path $TargetPaper 'numeric_reviews.json'), '--repo-root', $TargetRepo) + $realArg) | Out-Null
      }
    }

    Invoke-Step 'receipts' {
      Wait-Memory
      $r1 = Join-Path $OutRoot 'p15_apply_before_migration.json'
      $code = Invoke-Native -Exe $Python -Arguments @("$GateTools\p15_verify_numbers.py", '--paper-dir', $TargetPaper, '--stats-dir', $StatsDir,
                                                       '--review-file', (Join-Path $TargetPaper 'numeric_reviews.json'), '--report', $r1) -Name 'p15_before_migration' -TailLines 4
      if (-not (Test-Path $r1)) { Fail 'p15 wrote no report' }
      $migrate = @("$Tools\cr_migrate_receipts.py", '--paper-dir', $TargetPaper, '--old-dir', $Snapshot, '--results-dir', $ResultsDir, '--report', $r1,
                   '--out', (Join-Path $OutRoot 'receipt_migration.json'), '--apply') + $realArg
      if ($AcceptReceipts) {
        if (-not $FigureReviewer.Trim()) { Fail '-AcceptReceipts needs -FigureReviewer (the human who checked them)' }
        $migrate += @('--accept') + $AcceptReceipts + @('--reviewer', $FigureReviewer)
      }
      Invoke-Py 'migrate_receipts' $migrate | Out-Null
      $r2 = Join-Path $OutRoot 'p15_apply.json'
      $code2 = Invoke-Native -Exe $Python -Arguments @("$GateTools\p15_verify_numbers.py", '--paper-dir', $TargetPaper, '--stats-dir', $StatsDir,
                                                        '--review-file', (Join-Path $TargetPaper 'numeric_reviews.json'), '--report', $r2) -Name 'p15_after_migration' -TailLines 4
      $rep = Read-Json $r2
      if ($code2 -ne 0) {
        Write-Host 'p15 blockers. For a literal listed as manual with a suggested predecessor in receipt_migration.json, read the old receipt against the new context and re-run -ApplyToPaper -AcceptReceipts <id>; otherwise write a context-hashed receipt:' -ForegroundColor Yellow
        @(Get-Prop $rep 'unresolved_ids') | Where-Object { $_ } | Select-Object -First 40 | ForEach-Object { Write-Host "  unresolved: $_" }
        @(Get-Prop $rep 'errors') | Where-Object { $_ } | Select-Object -First 40 | ForEach-Object { Write-Host "  error: $_" }
        Fail "p15 does not pass after the receipt migration (report $r2)" 4
      }
      "p15 before migration exit $code; after migration PASS"
    }

    if ($Rehearsal -and $RehearsalFillAcknowledgments) {
      Invoke-Step 'rehearsal-acknowledgment' { Invoke-Py 'rehearsal_ack' @("$Tools\cr_postfreeze.py", 'rehearsal-ack', '--paper-dir', $TargetPaper) | Out-Null }
    }

    $release = @("$GateTools\p13_build_zip.py", '--camera-ready', '--stats-dir', $StatsDir, '--citation-record', $CitationRecord,
                 '--review-file', (Join-Path $TargetPaper 'numeric_reviews.json'))
    if ($Rehearsal) {
      $rel = Join-Path $OutRoot 'release'
      New-Item -ItemType Directory -Force -Path $rel | Out-Null
      $release += @('--paper-dir', $TargetPaper, '--out', "$rel\cr_rehearsal.zip", '--pdf-out', "$rel\cr_rehearsal.pdf",
                    '--docx-out', "$rel\cr_rehearsal.docx", '--manifest-out', "$rel\cr_rehearsal.release.json", '--staging-root', "$OutRoot\release_work")
    } elseif ($ReleaseOut) {
      $release += @('--out', $ReleaseOut)
    }
    if ($SkipRelease) {
      Write-Host "Release command (not executed, -SkipRelease): $Python $(ConvertTo-ArgString $release)"
    } else {
      Invoke-Step 'p13-camera-ready' {
        Wait-Memory
        Wait-Tectonic
        # Idle from creation, so p13's own Tectonic child inherits Idle (V8-6).
        Invoke-Py 'p13_camera_ready' $release -WorkDir $TargetRepo -Priority 'Idle' | Out-Null
        'ALL_PASS'
      }
    }
    Write-Host ''
    Write-Host 'Applied. Next: review git diff (paper, numeric_reviews.json, release_assets.json), commit, sync per PAPER_STATE.md.' -ForegroundColor Yellow
  }
} catch {
  if ($ExitCode -eq 0) { $ExitCode = 1 }
  Write-Host "PIPELINE STOPPED: $($_.Exception.Message)" -ForegroundColor Red
  if (-not $ApplyToPaper -and (Test-Path (Join-Path $OutRoot 'plan.json'))) {
    try { & $Python "$Tools\cr_postfreeze.py" summary --out-root $OutRoot | Out-Null } catch { }
  }
} finally {
  Write-Host ("steps: {0}" -f $StepsFile)
  Stop-Transcript | Out-Null
}
exit $ExitCode
