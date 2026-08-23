# Autopilot resource monitor - samples GPU/RAM/disk and enforces thermal guard.
# Writes timestamped rows to RESOURCE_MONITOR.csv and alerts to RESOURCE_ALERTS.log
param(
    [int]$IntervalSec = 45,
    [string]$OutDir = "C:\Users\Gary\Desktop\jepa\autopilot"
)

$csv    = Join-Path $OutDir "RESOURCE_MONITOR.csv"
$alerts = Join-Path $OutDir "RESOURCE_ALERTS.log"

if (-not (Test-Path $csv)) {
    "timestamp,gpu_util_pct,gpu_temp_c,vram_used_mib,vram_total_mib,vram_pct,ram_used_gb,ram_total_gb,ram_pct,cpu_pct,c_free_gb,c_free_pct,d_free_gb,d_free_pct,gpu_procs" | Out-File -FilePath $csv -Encoding utf8
}

# Safety thresholds (conservative defaults, per operator directive)
$VRAM_WARN = 85; $VRAM_STOP = 90
$RAM_WARN  = 80; $RAM_STOP  = 85
$DISK_WARN = 15; $DISK_STOP = 10
$TEMP_HOT  = 84
$hotSince  = $null

function Write-Alert($level, $msg) {
    $line = "$(Get-Date -Format o) [$level] $msg"
    Add-Content -Path $alerts -Value $line
}

while ($true) {
    try {
        $ts = Get-Date -Format o

        $g = (nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits) -split ','
        $gutil = [int]$g[0].Trim(); $gtemp = [int]$g[1].Trim()
        $vused = [int]$g[2].Trim(); $vtot = [int]$g[3].Trim()
        $vpct  = [math]::Round(100.0 * $vused / $vtot, 1)

        $gp = (nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits) 2>$null
        $gprocs = if ($gp) { ($gp | ForEach-Object { ($_ -split ',')[0].Trim() }) -join '|' } else { "none" }

        $os = Get-CimInstance Win32_OperatingSystem
        $ramTot = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        $ramFree = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $ramUsed = [math]::Round($ramTot - $ramFree, 2)
        $rampct = [math]::Round(100.0 * $ramUsed / $ramTot, 1)

        $cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 1)

        $cd = Get-PSDrive C; $dd = Get-PSDrive D
        $cfree = [math]::Round($cd.Free / 1GB, 2)
        $cpct  = [math]::Round(100.0 * $cd.Free / ($cd.Free + $cd.Used), 1)
        $dfree = [math]::Round($dd.Free / 1GB, 2)
        $dpct  = [math]::Round(100.0 * $dd.Free / ($dd.Free + $dd.Used), 1)

        "$ts,$gutil,$gtemp,$vused,$vtot,$vpct,$ramUsed,$ramTot,$rampct,$cpu,$cfree,$cpct,$dfree,$dpct,$gprocs" |
            Add-Content -Path $csv

        if ($vpct -ge $VRAM_STOP) { Write-Alert "STOP"  "VRAM at $vpct% (>= $VRAM_STOP). Do not launch further GPU processes." }
        elseif ($vpct -ge $VRAM_WARN) { Write-Alert "WARN" "VRAM at $vpct% (>= $VRAM_WARN)." }

        if ($rampct -ge $RAM_STOP) { Write-Alert "STOP" "System RAM at $rampct% (>= $RAM_STOP). Serialize memory-heavy work." }
        elseif ($rampct -ge $RAM_WARN) { Write-Alert "WARN" "System RAM at $rampct% (>= $RAM_WARN)." }

        if ($cpct -lt $DISK_STOP -or $cfree -lt 25) { Write-Alert "STOP" "C: free ${cfree}GB / ${cpct}% - below stop threshold. Route large outputs to D:." }
        elseif ($cpct -lt $DISK_WARN) { Write-Alert "WARN" "C: free ${cfree}GB / ${cpct}%." }
        if ($dpct -lt $DISK_WARN) { Write-Alert "WARN" "D: free ${dfree}GB / ${dpct}%." }

        if ($gtemp -ge $TEMP_HOT) {
            if ($null -eq $hotSince) { $hotSince = Get-Date; Write-Alert "WARN" "GPU at ${gtemp}C (>= ${TEMP_HOT}C). Starting 2-minute hot timer." }
            elseif (((Get-Date) - $hotSince).TotalSeconds -ge 120) {
                Write-Alert "THERMAL" "GPU sustained ${gtemp}C for >=120s. Operator policy: pause/stop active GPU job, preserve state, allow cooling, diagnose."
            }
        } else { if ($null -ne $hotSince) { Write-Alert "INFO" "GPU cooled to ${gtemp}C. Hot timer cleared." }; $hotSince = $null }
    }
    catch { Write-Alert "ERROR" "monitor exception: $($_.Exception.Message)" }

    Start-Sleep -Seconds $IntervalSec
}
