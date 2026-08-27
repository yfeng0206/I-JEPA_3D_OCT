# Disk Cleanup - Phase D1 (SAFE / low-risk targets)

Date: 2026-08-26
Machine: DESKTOP-CJETT24
Executed elevated (Administrator): yes

## Starting state (measured at run time)

| Drive | Used GB | Free GB | Percent free |
|---|---|---|---|
| C: | 215.01 | 16.32 | 7.06 |

The brief quoted 214.9 GB used / 16.4 GB free; the live measurement at run start was
215.01 / 16.32. Normal OS churn between the two measurements.

## Results

All "GB reclaimed" values are the measured delta in `Get-PSDrive C` free space taken
immediately before and immediately after each step, so they include a small amount of
unrelated OS write activity (the operator's machine was in use).

| Target | GB before | GB reclaimed | Action taken | Notes / skips |
|---|---|---|---|---|
| C:\ProgramData\anaconda3 (conda package cache) | 21.114 | 0.000 | Ran `C:\ProgramData\anaconda3\Scripts\conda.exe clean --all --yes` | conda reported: no unused tarballs, no index cache, no unused packages, no tempfiles, no logfiles. Nothing to reclaim. Directory still 21.114 GB. Anaconda itself NOT uninstalled (out of scope for D1). |
| C:\Users\Gary\AppData\Local\Temp | 1.227 | 1.238 | Deleted all 540 top-level entries | 4 entries locked/in-use and skipped (expected). Residual after: 0.009 GB. |
| C:\$WINDOWS.~BT | 0.543 | see below | takeown /r + icacls grant Administrators:F, then Remove-Item -Recurse -Force | Removed; folder no longer present. |
| C:\$GetCurrent | 0.000 | see below | Same as above | Removed; folder no longer present. It measured 0 GB (metadata only). |
| C:\$SysReset | 0.052 | see below | Same as above | Removed; folder no longer present. |
| (the three Windows upgrade leftovers, combined) | 0.595 | 0.258 | - | Measured free-space gain was less than the sum of file sizes. Expected: a large share of the files under $WINDOWS.~BT are NTFS hard links into WinSxS, so deleting the directory entry does not release those bytes. |
| Recycle Bin (C:) | 0.161 | 0.161 | `Clear-RecycleBin -DriveLetter C -Force` | Clean; verified 0 GB afterwards. |
| pip cache (C:\Users\Gary\AppData\Local\pip\cache) | 0.019 | see below | `pip cache purge` -> "Files removed: 20" | Per-user cache, shared by the D:\jepa_phase0\.venv interpreter as well. Fully regenerable. |
| npm cache | MISSING | 0.000 | None | SKIPPED: npm is not installed and C:\Users\Gary\AppData\Local\npm-cache does not exist. |
| tectonic caches | 0.091 | see below | Removed C:\Users\Gary\AppData\Local\TectonicProject\Tectonic\Cache and C:\Users\Gary\AppData\Local\tectonic | Two separate cache roots found (0.043 GB + 0.048 GB). Both are download caches for TeX resources and are regenerated on next tectonic run. |
| (pip + tectonic, combined) | 0.110 | 0.111 | - | - |

### Anaconda on PATH - findings

Requested check: does anything on PATH resolve into C:\ProgramData\anaconda3?

- `conda` is NOT on PATH. The executable exists at C:\ProgramData\anaconda3\Scripts\conda.exe
  but is only reachable by absolute path.
- No PATH entry contains "anaconda" or "conda".
- `python`  -> C:\Users\Gary\AppData\Local\Programs\Python\Python311\python.exe
- `pip`     -> C:\Users\Gary\AppData\Local\Programs\Python\Python311\Scripts\pip.exe
- `pythonw` -> C:\Users\Gary\AppData\Local\Programs\Python\Python311\pythonw.exe
- `jupyter` -> not found
- `ipython` -> not found

Conclusion: nothing on PATH resolves to the anaconda install. C:\ProgramData\anaconda3
holds 21.114 GB with 575 package directories under \pkgs, all of which conda considers
in-use by the base environment. It is the single largest recoverable item on C:, but per
the D1 brief it was NOT uninstalled. Recommend evaluating an anaconda uninstall in a
later phase with operator consent.

## Reported but deliberately NOT actioned

| Item | GB | Reason |
|---|---|---|
| C:\hiberfil.sys | 12.77 | Available to reclaim, but requires `powercfg /hibernate off`, which disables hibernate and fast startup. Per instructions this needs the operator's explicit consent. NOT touched, no powercfg run. |
| C:\ProgramData\anaconda3 (whole install) | 21.114 | Out of scope for D1. Nothing on PATH uses it (see above), so it is a strong candidate for a later phase. |
| C:\Windows\Temp | 0.386 | Not on the D1 target list. Reported rather than deleted, per the "prefer reporting when ambiguous" rule. |
| System Restore / shadow copies | unknown | Excluded by instruction. vssadmin was NOT run. |
| C:\jepa_data | 70.8 | Excluded: active training data. Verified still present, including \slice_cache. |
| D:\jepa_phase0 | - | Excluded: checkpoints and cached features. Verified still present. |
| C:\Users\Gary\Desktop\jepa | - | Excluded: git repo. Verified still present. |
| Games / launchers (C:\SteamLibrary, C:\GOG Games, D:\Riot Games, D:\XboxGames, D:\Battle.net, D:\WeGameApps, AppData\Local\Google, \Roblox, \Discord, \Steam) | - | Excluded by instruction. C:\SteamLibrary verified still present. |

## Safety compliance

- No process was stopped. `Stop-Process` was never invoked.
- No service was disabled.
- No `powercfg` run. No `vssadmin` run.
- Locked files were skipped, never forced (4 in user Temp).
- Work was sequential and single-threaded to keep CPU and disk load modest.
- Post-run existence check passed for every protected path listed above.

## Final state

| Drive | Used GB | Free GB | Percent free |
|---|---|---|---|
| C: | 213.29 | 18.04 | 7.80 |

Net reclaimed this phase: 1.72 GB (16.32 GB free -> 18.04 GB free).
Percent free improved from 7.06 to 7.80.

The phase completed without incident, but the yield was limited: the largest listed
target (anaconda, 21.1 GB) produced nothing from cache cleaning alone, because its disk
use is the installed packages themselves rather than a reclaimable cache. Meaningful
relief on C: will require a later phase decision on anaconda (21.1 GB) and/or hibernation
(12.8 GB), both of which need operator consent.
