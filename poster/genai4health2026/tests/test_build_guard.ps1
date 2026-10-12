<#
.SYNOPSIS
  Regression tests for build.ps1's synthetic-output guard: -AllowSynthetic must be
  refused inside any Git repository or worktree, also when it is reached through a
  junction or symbolic link, with or without git on PATH, in PowerShell 7 and
  Windows PowerShell 5.1.
.EXAMPLE
  pwsh -NoProfile -File tests\test_build_guard.ps1 -Work C:\t\guard_tests
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_build_guard.ps1 -Work C:\t\guard_tests
.NOTES
  Each case builds a tree <...>\poster\genai4health2026 (a copy of build.ps1 and
  make_preview.py) with a sibling paper\genai4health2026\auto holding only a synthetic
  marker file, so nothing is ever compiled: when the guard refuses, build.ps1 stops
  before the results check; when it allows, the results check runs (and rejects the
  incomplete bundle). Repositories are real `git init` trees when git is available
  (a .git folder otherwise). Symbolic-link cases are reported as skipped if this
  account may not create symbolic links. Files are written only under -Work; links
  are removed as links before any folder is deleted.
#>
param([Parameter(Mandatory)] [string]$Work)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$poster = Split-Path -Parent $here
$Work = [IO.Path]::GetFullPath($Work)
$git = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1

# Remove links (junctions, symbolic links) as links first, never following them.
function Remove-Links([string]$Dir) {
  foreach ($item in (New-Object IO.DirectoryInfo $Dir).GetFileSystemInfos()) {
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
      if ($item -is [IO.DirectoryInfo]) { [IO.Directory]::Delete($item.FullName) }
      else { [IO.File]::Delete($item.FullName) }
    } elseif ($item -is [IO.DirectoryInfo]) {
      Remove-Links $item.FullName
    }
  }
}
function Remove-TreeSafely([string]$Dir) {
  if (Test-Path -LiteralPath $Dir) {
    Remove-Links $Dir
    Remove-Item -LiteralPath $Dir -Recurse -Force
  }
}

# Run a native command without letting its stderr become a terminating error
# (Windows PowerShell 5.1); returns @(exit code, text).
function Invoke-Child([string]$Exe, [string[]]$Arguments) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $text = (& $Exe @Arguments 2>&1 | ForEach-Object { "$_" }) -join "`n"
    return @($LASTEXITCODE, $text)
  } finally {
    $ErrorActionPreference = $saved
  }
}

function New-Repo([string]$Dir) {
  New-Item -ItemType Directory -Force $Dir | Out-Null
  if ($git) {
    $r = Invoke-Child $git.Path @('init', '--quiet', $Dir)
    if ($r[0] -ne 0) { throw "git init failed in $Dir" }
  } else {
    New-Item -ItemType Directory (Join-Path $Dir '.git') | Out-Null
  }
}

function New-Tree([string]$Root) {
  $p = Join-Path $Root 'poster\genai4health2026'
  $a = Join-Path $Root 'paper\genai4health2026\auto'
  New-Item -ItemType Directory -Force $p, $a | Out-Null
  Copy-Item (Join-Path $poster 'build.ps1'), (Join-Path $poster 'make_preview.py') $p
  Set-Content -Encoding ascii (Join-Path $a 'cr_results.tex') '% SYNTHETIC DRY RUN: these are not results.'
  return $p
}

function New-Link([string]$Kind, [string]$Path, [string]$Target) {
  New-Item -ItemType $Kind -Path $Path -Value $Target | Out-Null
}

Remove-TreeSafely $Work
New-Item -ItemType Directory -Force $Work | Out-Null
$noGitPath = (($env:PATH -split ';') | Where-Object { $_ -and -not (Test-Path (Join-Path $_ 'git.exe')) }) -join ';'

# Each case: setup($root) returns the poster folder to build from.
$cases = @(
  @{ name = 'outside_repository'; expect = 'allowed'; setup = { param($r) New-Tree $r } },
  @{ name = 'git_repository'; expect = 'refused'; setup = { param($r) New-Repo $r; New-Tree $r } },
  @{ name = 'worktree_git_file'; expect = 'refused'; setup = { param($r) Set-Content -Encoding ascii (Join-Path $r '.git') 'gitdir: C:/elsewhere/.git/worktrees/poster'; New-Tree $r } },
  @{ name = 'git_repository_in_parent'; expect = 'refused'; setup = { param($r) New-Repo $r; New-Tree (Join-Path $r 'nested\deeper') } },
  @{ name = 'git_repository_git_missing'; expect = 'refused'; nogit = $true; setup = { param($r) New-Repo $r; New-Tree $r } },
  @{ name = 'worktree_file_git_missing'; expect = 'refused'; nogit = $true; setup = { param($r) Set-Content -Encoding ascii (Join-Path $r '.git') 'gitdir: C:/elsewhere/.git/worktrees/poster'; New-Tree $r } },
  @{ name = 'outside_repository_git_missing'; expect = 'allowed'; nogit = $true; setup = { param($r) New-Tree $r } },
  @{ name = 'GIT_DIR_set'; expect = 'refused'; gitdir = $true; setup = { param($r) New-Tree $r } },
  @{ name = 'results_folder_uninspectable'; expect = 'refused'; setup = { param($r) $p = New-Tree $r; Remove-Item -Recurse -Force (Join-Path $r 'paper'); $p } },
  # V9: a junction alias of a repository subtree has no lexical .git ancestor.
  @{ name = 'junction_alias_into_repo'; expect = 'refused'; setup = { param($r) New-Repo "$r\repo"; New-Tree "$r\repo\subtree" | Out-Null; New-Link Junction "$r\alias" "$r\repo\subtree"; "$r\alias\poster\genai4health2026" } },
  @{ name = 'junction_alias_into_repo_git_missing'; expect = 'refused'; nogit = $true; setup = { param($r) New-Repo "$r\repo"; New-Tree "$r\repo\subtree" | Out-Null; New-Link Junction "$r\alias" "$r\repo\subtree"; "$r\alias\poster\genai4health2026" } },
  @{ name = 'junction_ancestor_into_repo'; expect = 'refused'; setup = { param($r) New-Repo "$r\repo"; New-Tree "$r\repo\a\b" | Out-Null; New-Link Junction "$r\alias" "$r\repo\a"; "$r\alias\b\poster\genai4health2026" } },
  @{ name = 'junction_results_into_repo'; expect = 'refused'; setup = { param($r) $p = New-Tree "$r\tree"; New-Repo "$r\repo"; Move-Item "$r\tree\paper" "$r\repo\paper"; New-Link Junction "$r\tree\paper" "$r\repo\paper"; $p } },
  @{ name = 'junction_output_folder_into_repo'; expect = 'refused'; setup = { param($r) $p = New-Tree "$r\tree"; New-Repo "$r\repo"; New-Item -ItemType Directory "$r\repo\out" | Out-Null; New-Link Junction "$p\generated" "$r\repo\out"; $p } },
  @{ name = 'junction_alias_outside_repo'; expect = 'allowed'; setup = { param($r) New-Tree "$r\plain" | Out-Null; New-Link Junction "$r\alias" "$r\plain"; "$r\alias\poster\genai4health2026" } },
  @{ name = 'symlink_alias_into_repo'; expect = 'refused'; symlink = $true; setup = { param($r) New-Repo "$r\repo"; New-Tree "$r\repo\subtree" | Out-Null; New-Link SymbolicLink "$r\alias" "$r\repo\subtree"; "$r\alias\poster\genai4health2026" } },
  @{ name = 'symlink_output_file_into_repo'; expect = 'refused'; symlink = $true; setup = { param($r) $p = New-Tree "$r\tree"; New-Repo "$r\repo"; Set-Content -Encoding ascii "$r\repo\poster.pdf" 'x'; New-Link SymbolicLink "$p\poster.pdf" "$r\repo\poster.pdf"; $p } }
)

$failures = 0
$skipped = 0
foreach ($shell in 'pwsh', 'powershell') {
  foreach ($c in $cases) {
    $root = Join-Path $Work "$shell\$($c.name)"
    New-Item -ItemType Directory -Force $root | Out-Null
    try {
      $p = @(& $c.setup $root)[-1]
    } catch {
      if ($c.symlink) {
        $skipped++
        "{0,-10} {1,-38} skipped: cannot create symbolic links here ({2})" -f $shell, $c.name, $_.Exception.Message.Split("`n")[0]
        continue
      }
      throw
    }
    $exe = if ($shell -eq 'pwsh') { 'pwsh' } else { 'powershell.exe' }
    $shellArgs = @('-NoProfile')
    if ($shell -eq 'powershell') { $shellArgs += @('-ExecutionPolicy', 'Bypass') }
    $shellArgs += @('-File', (Join-Path $p 'build.ps1'), '-OnlyCached', '-AllowSynthetic')
    $savedPath, $savedGitDir = $env:PATH, $env:GIT_DIR
    try {
      if ($c.nogit) { $env:PATH = $noGitPath }
      if ($c.gitdir) { $env:GIT_DIR = 'C:\elsewhere\.git' }
      $r = Invoke-Child $exe $shellArgs
    } finally {
      $env:PATH = $savedPath
      $env:GIT_DIR = $savedGitDir
    }
    $rc, $out = $r[0], $r[1]
    $refused = $out -match 'AllowSynthetic is refused'
    $reachedCheck = $out -match 'Precheck'
    $native = $out -match 'NativeCommandError'
    $got = if ($refused -and -not $reachedCheck) { 'refused' } elseif ($reachedCheck -and -not $refused) { 'allowed' } else { 'unclear' }
    $compiled = Test-Path -LiteralPath (Join-Path $p 'poster.pdf') -PathType Leaf
    if ($c.name -eq 'symlink_output_file_into_repo') { $compiled = ((Get-Content -Raw "$root\repo\poster.pdf").Trim() -ne 'x') }
    $ok = ($got -eq $c.expect) -and ($rc -ne 0) -and -not $native -and -not $compiled
    if (-not $ok) { $failures++ }
    $note = if ($c.nogit) { 'git not on PATH' } else { '' }
    "{0,-10} {1,-38} expect {2,-8} got {3,-8} exit {4} {5} {6}" -f $shell, $c.name, $c.expect, $got, $rc, $(if ($ok) { 'ok' } else { 'FAIL' }), $note
    if (-not $ok) { $out }
  }
}
Remove-TreeSafely $Work
$summary = if ($failures) { "$failures case(s) failed" } else { 'all cases behave as expected' }
if ($skipped) { $summary += "; $skipped symbolic-link case(s) skipped" }
$summary
exit $failures
