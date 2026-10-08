<#
.SYNOPSIS
  Builds the GenAI4Health 2026 poster: design master (poster.pdf), exact-size
  print PDF (poster_print.pdf) and a headless PNG preview, then prints a
  build report (page count, size, overfull boxes, fit check, placeholders).
.PARAMETER Size
  Optional one-off override of poster_size.tex: neurips-ws | a0 | a1 | us-36x48.
  Outputs are then named poster-<size>.pdf, poster_print-<size>.pdf and
  poster-<size>_preview.png.
.PARAMETER Dpi
  PNG preview resolution (default 50; 100 gives an upload-quality PNG).
.PARAMETER OnlyCached
  Pass --only-cached to Tectonic (fully offline after one online build).
.NOTES
  Tectonic runs at Idle priority, one compile at a time. Override the
  executable with $env:TECTONIC.
#>
param(
  [ValidateSet('', 'neurips-ws', 'a0', 'a1', 'us-36x48')] [string]$Size = '',
  [int]$Dpi = 50,
  [switch]$OnlyCached
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tectonic = if ($env:TECTONIC) { $env:TECTONIC } else { 'D:\jepa_phase0\tools\tectonic\tectonic.exe' }
$env:OMP_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'

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

$rc = 1
Push-Location $here
try {
  New-Item -ItemType Directory -Force build_logs | Out-Null
  $running = Get-Process -Name tectonic -ErrorAction SilentlyContinue
  if ($running) {
    throw "Another Tectonic process is running (PID $($running.Id -join ', ')); run one compile at a time."
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
  try {
    Invoke-Tectonic "$design.tex"
    Invoke-Tectonic "$print.tex"
  } finally {
    if ($Size) { Remove-Item "$design.tex", "$print.tex" -ErrorAction SilentlyContinue }
  }
  python make_preview.py "$print.pdf" --design-log "$design.log" --png "${design}_preview.png" --dpi $Dpi
  $rc = $LASTEXITCODE
} finally {
  Pop-Location
}
exit $rc
