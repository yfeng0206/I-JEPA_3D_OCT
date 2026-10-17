<#
.SYNOPSIS
  Builds the GenAI4Health 2026 poster: validates the registered-replication results,
  then builds the design master (poster.pdf), the exact-size print PDF
  (poster_print.pdf) and a headless PNG preview, and prints a build report (page
  count, size, overfull boxes, fit check, placeholders, result cells checked
  against the generator's metadata).
.PARAMETER Size
  Optional one-off override of poster_size.tex: neurips26-ws | neurips-ws | a0 | a1 | us-36x48.
  Outputs are then named poster-<size>.pdf, poster_print-<size>.pdf and
  poster-<size>_preview.png.
.PARAMETER Dpi
  PNG preview resolution (default 50; 100 gives an upload-quality PNG).
.PARAMETER OnlyCached
  Pass --only-cached to Tectonic (fully offline after one online build).
.PARAMETER AllowSynthetic
  Accept SYNTHETIC DRY RUN results in ../../paper/genai4health2026/auto
  (generator tests). Refused when this folder, the results folder or any existing
  output/log item here is, or leads through a junction or symbolic link, inside a
  Git repository or worktree (a .git directory or .git file in any parent of the
  lexical or the resolved physical path, or GIT_DIR/GIT_WORK_TREE set), or when
  that cannot be determined, so synthetic output can only be built in a scratch
  copy. Git itself is not needed.
.NOTES
  Works in PowerShell 7 and Windows PowerShell 5.1. Tectonic runs at Idle
  priority, one compile at a time; override it with $env:TECTONIC. Python:
  $env:POSTER_PYTHON, else D:\jepa_phase0\.venv\Scripts\python.exe if present,
  else python on PATH; it runs CPU-only.
#>
param(
  [ValidateSet('', 'neurips26-ws', 'neurips-ws', 'a0', 'a1', 'us-36x48')] [string]$Size = '',
  [int]$Dpi = 50,
  [switch]$OnlyCached,
  [switch]$AllowSynthetic
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tectonic = if ($env:TECTONIC) { $env:TECTONIC } else { 'D:\jepa_phase0\tools\tectonic\tectonic.exe' }
$python = if ($env:POSTER_PYTHON) { $env:POSTER_PYTHON }
          elseif (Test-Path -LiteralPath 'D:\jepa_phase0\.venv\Scripts\python.exe') { 'D:\jepa_phase0\.venv\Scripts\python.exe' }
          else { 'python' }
$env:OMP_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONWARNINGS = 'ignore'

function Invoke-Tectonic([string]$TexFile) {
  $stem = [IO.Path]::GetFileNameWithoutExtension($TexFile)
  $targs = @($TexFile, '--keep-logs', '--chatter', 'minimal')
  if ($OnlyCached) { $targs += '--only-cached' }
  $p = Start-Process -FilePath $tectonic -ArgumentList $targs -NoNewWindow -PassThru `
    -RedirectStandardOutput "build_logs\$stem.stdout.txt" `
    -RedirectStandardError "build_logs\$stem.stderr.txt"
  $null = $p.Handle
  try { $p.PriorityClass = 'Idle' } catch { }
  $p.WaitForExit()
  if ($p.ExitCode -ne 0) {
    Get-Content "build_logs\$stem.stderr.txt" -Tail 30
    throw "Tectonic failed on $TexFile (exit $($p.ExitCode)); see $stem.log"
  }
}

# Python's stderr must not become a terminating NativeCommandError (Windows
# PowerShell 5.1); print every line and return the exit code.
function Invoke-Python([string[]]$PyArgs) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $python @PyArgs 2>&1 | ForEach-Object { Write-Host "$_" }
    return $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $saved
  }
}

# Physical path of an existing file or folder: Windows resolves every junction,
# symbolic link and mount point on the way (GetFinalPathNameByHandle). Works in
# PowerShell 7 and Windows PowerShell 5.1; throws if the path cannot be resolved.
function Get-PhysicalPath([string]$Path) {
  if (-not ('PosterFinalPath' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;
public static class PosterFinalPath {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  private static extern SafeFileHandle CreateFileW(string name, uint access, uint share,
      IntPtr security, uint disposition, uint flags, IntPtr template);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  private static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder path,
      uint size, uint flags);
  public static string Resolve(string path) {
    // 0x02000000 FILE_FLAG_BACKUP_SEMANTICS (folders); reparse points are followed.
    SafeFileHandle handle = CreateFileW(path, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero);
    try {
      if (handle.IsInvalid) { throw new IOException("cannot open " + path); }
      StringBuilder buffer = new StringBuilder(32768);
      uint n = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 0);
      if (n == 0 || n >= buffer.Capacity) { throw new IOException("cannot resolve " + path); }
      string s = buffer.ToString();
      if (s.StartsWith(@"\\?\UNC\")) { return @"\\" + s.Substring(8); }
      if (s.StartsWith(@"\\?\")) { return s.Substring(4); }
      return s;
    } finally {
      handle.Dispose();
    }
  }
}
'@
  }
  return [PosterFinalPath]::Resolve([IO.Path]::GetFullPath($Path))
}

# True if a .git directory or .git file (worktree, submodule) is in the folder
# $FullPath (or the folder of a file) or any parent; true if a folder on the way
# does not exist or cannot be listed.
function Test-GitAncestor([string]$FullPath) {
  $dir = New-Object IO.DirectoryInfo $FullPath
  if ([IO.File]::Exists($FullPath)) { $dir = $dir.Parent }
  while ($null -ne $dir) {
    if (-not $dir.Exists) { return $true }
    if ([IO.Directory]::GetFileSystemEntries($dir.FullName, '.git').Count -gt 0) { return $true }
    $dir = $dir.Parent
  }
  return $false
}

# True if $Path is inside a Git repository or worktree, or if that cannot be
# determined (fail closed): checks the lexical path and its physical path (all
# junctions and symbolic links resolved) and every parent of both. Git itself is
# not needed; GIT_DIR/GIT_WORK_TREE set also counts as inside.
function Test-InsideRepository([string]$Path) {
  if ($env:GIT_DIR -or $env:GIT_WORK_TREE) { return $true }
  try {
    $lexical = [IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $lexical)) { return $true }
    $physical = Get-PhysicalPath $lexical
    return (Test-GitAncestor $lexical) -or (Test-GitAncestor $physical)
  } catch {
    return $true
  }
}

# Every place a synthetic build reads results from or writes outputs to: this
# folder, the results folder, and every existing item (output files, logs, links)
# in this folder and in its build_logs and generated folders.
function Get-GuardedPaths([string]$PosterDir, [string]$ResultsDir) {
  $paths = @($PosterDir, $ResultsDir)
  foreach ($dir in @($PosterDir, (Join-Path $PosterDir 'build_logs'), (Join-Path $PosterDir 'generated'))) {
    if (Test-Path -LiteralPath $dir) {
      $paths += @(Get-ChildItem -LiteralPath $dir -Force | ForEach-Object { $_.FullName })
    }
  }
  return $paths
}

$rc = 1
Push-Location $here
try {
  # Results gate: validate auto/ and write generated/results_state.* before
  # compiling, so invalid or synthetic results never reach the tracked outputs.
  $pre = @('make_preview.py', '--precheck')
  if ($AllowSynthetic) {
    $resultsDir = Join-Path $here '..\..\paper\genai4health2026\auto'
    foreach ($place in (Get-GuardedPaths $here $resultsDir)) {
      if (Test-InsideRepository $place) {
        throw "-AllowSynthetic is refused: '$place' is or leads (junction/link) inside a Git repository or worktree, or its location cannot be resolved; build synthetic results in a scratch copy outside any repository."
      }
    }
    $pre += '--allow-synthetic'
  }
  New-Item -ItemType Directory -Force build_logs | Out-Null
  if ((Invoke-Python $pre) -ne 0) {
    throw 'Results check failed (see PROBLEMS above); nothing was compiled.'
  }
  if ($Size) {
    $design = "poster-$Size"
    $print = "poster_print-$Size"
    Set-Content -Encoding ascii "$design.tex" "\def\PosterSize{$Size}\input{poster}"
    Set-Content -Encoding ascii "$print.tex" "\def\PosterSize{$Size}\def\PosterDesignPDF{$design.pdf}\input{poster_print}"
  } else {
    $design = 'poster'
    $print = 'poster_print'
  }
  $running = Get-Process -Name tectonic -ErrorAction SilentlyContinue
  if ($running) {
    if ($Size) { Remove-Item "$design.tex", "$print.tex" -ErrorAction SilentlyContinue }
    throw "Another Tectonic process is running (PID $($running.Id -join ', ')); run one compile at a time."
  }
  try {
    Invoke-Tectonic "$design.tex"
    Invoke-Tectonic "$print.tex"
  } finally {
    if ($Size) { Remove-Item "$design.tex", "$print.tex" -ErrorAction SilentlyContinue }
  }
  $chk = @('make_preview.py', "$print.pdf", '--design-log', "$design.log", '--png', "${design}_preview.png", '--dpi', "$Dpi")
  if ($AllowSynthetic) { $chk += '--allow-synthetic' }
  $rc = Invoke-Python $chk
} finally {
  Pop-Location
}
exit $rc
