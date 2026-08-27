# DISK_CLEANUP_FULL - Deep C: drive cleanup and full accounting

Host: DESKTOP-CJETT24
Volume: C: (disk 0, CT250MX500SSD1, 231.330 GB)
Run date: 2026-08-26
Elevation: Administrator (MEASURED)

## Headline

| Metric | Value | Basis |
| --- | --- | --- |
| C: free BEFORE | 17.228 GB (7.45 percent) | MEASURED |
| C: free AFTER | 29.572 GB (12.79 percent) | MEASURED |
| Total reclaimed | 12.344 GB | MEASURED (Win32_LogicalDisk delta) |
| Free space multiple | 1.72x more free space | MEASURED |

Nothing on the do-not-touch list was modified. No service was stopped or disabled.
No driver was touched. `powercfg` was not run. No process was killed.

---

## A. REMOVED

All "GB reclaimed" figures are MEASURED as the `Win32_LogicalDisk.FreeSpace` delta
immediately before and after that step, not as the logical size of the directory.
Where the two differ the discrepancy is explained in the notes.

| # | Item | GB reclaimed | Why it was safe |
| --- | --- | --- | --- |
| 1 | `C:\Users\Gary\.copilot\logs` - 10 Copilot CLI `process-*.log` files | 5.956 | Diagnostic transcript logs only, all dated 2026-08 (MEASURED: 8.150 GB logical, single largest file 3.34 GB). Session resume does NOT read these; it reads `session-state\` and `session-store.db`, both of which were RETAINED. Regenerates automatically on the next CLI run. All 10 deleted, zero locked. Logical 8.150 vs measured 5.956 because one 2.24 GB file was still held open by a live process, so its blocks return to the free pool when that handle closes (INFERRED). |
| 2 | WinSxS component store cleanup, pass 1 - `dism /online /cleanup-image /startcomponentcleanup` | 4.875 | DISM itself reported `Component Store Cleanup Recommended : Yes` with `Number of Reclaimable Packages : 10` and 9.21 GB of "Backups and Disabled Features" (MEASURED, pre-run analyze). Removes superseded component versions. `/ResetBase` was deliberately NOT used, so the ability to uninstall installed updates is preserved. |
| 3 | `C:\Windows\Temp` + `C:\Users\Gary\AppData\Local\Temp` | 0.465 | Scratch space by definition; anything still needed is recreated by its owner. 99 entries removed, 20 locked entries skipped without error per the safety rule. |
| 4 | Delivery Optimization peer cache (`Delete-DeliveryOptimizationCache -Force`) | 0.449 | Peer-to-peer update payload cache. Purely a re-download optimisation; Windows Update refills it on demand. Used the supported cmdlet, which clears the cache only - the DoSvc service was left running and enabled. Cache directory verified 0.000 GB afterwards (MEASURED). |
| 5 | `C:\Windows\Panther` Windows Setup upgrade logs | 0.382 | Setup/telemetry logs from the Nov 2025 in-place upgrade: `setupact.log` (0.582 GB), `monitor\` (0.499 GB), `MigLog.xml` (0.179 GB), APPRAISER/CompatData artefacts, `setup.etl`, `WinSetupMon.log`, `diagwrn.xml`. This is exactly the "Windows upgrade log files" Disk Cleanup target - diagnostic only. Rollback data, `UnattendGC`, `setup.exe` and `Resources` were left in place. Directory fell 1.35 GB -> 0.041 GB logical but only 0.382 GB was freed physically because `C:\Windows\Panther` carries the NTFS `Compressed` attribute (MEASURED via `(Get-Item).Attributes` = `Directory, Compressed, NotContentIndexed`). |
| 6 | WER report archive/queue, user CrashDumps, rotated CBS logs, Minidump, LiveKernelReports | 0.056 | 494 crash-report and rotated-log artefacts. Post-mortem diagnostics with no runtime role. The live `CBS.log` was explicitly excluded so the in-flight DISM transaction log stayed intact. WER subdirectories verified 0.000 GB afterwards (MEASURED). |
| 7 | WinSxS component store cleanup, pass 2 | 0.139 | Re-analysis after pass 1 still reported 2 reclaimable packages; a second pass collected them. Same safety argument as row 2. |
| 8 | `C:\Windows\SoftwareDistribution\Download` | 0.003 | Windows Update payload staging area. Already nearly empty. Windows re-downloads on demand. |
| | **TOTAL** | **12.344** | MEASURED end-to-end free-space delta. Row sum is 12.325; the 0.019 GB difference is background OS write churn during the run (INFERRED). |

Post-cleanup component store (MEASURED, `dism /analyzecomponentstore` re-run):
actual size fell from **17.78 GB to 12.23 GB**, and "Backups and Disabled Features"
fell from **9.21 GB to 3.68 GB**.

---

## B. RETAINED DESPITE BEING LARGE

| Item | GB | Specific reason retained |
| --- | --- | --- |
| `C:\jepa_data` (mirage_soft_guides 44.7 + slice_cache 26.1) | 70.78 | FLAGGED DO-NOT-TOUCH, and independently verified. `src\train_patch.py` lines 349-353 raise `FileNotFoundError("slice_cache_dir is set but %s does not exist. Build it with scripts/build_slice_cache.py.")` the moment the directory is absent (MEASURED - file read). Ten configs hard-code the `C:\jepa_data` path: `configs\patch_anatomy_v2.yaml:11,41`, `configs\patch_cover_ep25.yaml:11,47`, `configs\patch_cover_f021_ep25.yaml:11,47`, `configs\patch_cover_random_ep25.yaml:11,47`, plus six `configs\replication\rep_*.yaml:27` (MEASURED - grep). It sits on the SSD deliberately: `docs\experiments\masking\cover_random_campaign.md:116-117` records the guide cache and slice cache as "**SSD**, shared with blob", and line 329 states "**Moving caches to SSD.** Already there; the HDD holds only the raw volumes". D: is a spinning ST2000DM008, so relocating would throttle dataloader throughput on the paused-and-resumable pretraining replication. |
| `C:\hiberfil.sys` | 12.77 | FLAGGED. Disabling hibernation requires the operator's explicit consent. `powercfg` was NOT run. AVAILABLE WITH CONSENT: `powercfg /hibernate off` would free 12.77 GB (MEASURED file length) and take C: to roughly 42.3 GB free / 18.3 percent (INFERRED). This is the single largest remaining lever on C:. |
| `C:\ProgramData\anaconda3` | 21.11 | PROVEN IN USE - the prior pass's "nothing on PATH" finding is correct but incomplete. Confirmed absent from machine PATH, user PATH, all three Run keys, all four PowerShell profiles, and no venv on the machine derives from it (both `D:\jepa_phase0\.venv` and `C:\Users\Gary\Desktop\CopilotWorldLab\.venv` declare `home = ...\Python311` in `pyvenv.cfg`, MEASURED). BUT file-access forensics prove real execution: `Lib\site-packages\anaconda_navigator\...\__pycache__\*.pyc` were read on **2026-07-30 at 17:56**, and `C:\Windows\Prefetch\PYTHONW.EXE-7818B5CB.pf` was written **2026-07-30 17:57:01** - Anaconda Navigator was launched 27 days ago (MEASURED). A second import burst of 1,868 `.pyc` files occurred **2026-05-12 19:46** (MEASURED). `.pyc` reads are interpreter imports, not scanner activity; by contrast the 2026-08-12 05:52 touches hit only `.exe` stubs uniformly, which is the antivirus signature (MEASURED). A `TorchGPU` conda env (4.13 GB, CUDA 12.1 + PyTorch 2.2.1) is registered in `C:\Users\Gary\.conda\environments.txt`. The brief's condition "if and only if you can show it is genuinely unused" is NOT met - the evidence points the other way - so it was left intact. |
| `C:\Windows\Installer` | 12.14 | MSI/MSP installer cache. Windows requires these to repair, patch or uninstall every installed product; deleting them silently breaks future uninstalls and update rollbacks. No built-in tool distinguishes orphaned from live entries, and the brief says prefer reporting when ambiguous. Removing orphans needs a third-party tool and operator sign-off. |
| `C:\Windows\WinSxS` (residual) | 12.23 | Post-cleanup remainder. 8.57 GB is hard-linked into `C:\Windows\System32` and is not separately recoverable at all. The residual 3.68 GB of backups can only be reached with `/ResetBase`, which permanently blocks uninstalling installed updates - not done without consent. |
| `C:\Users\Gary\Desktop\CopilotWorldLab` | 6.85 | Operator's separate active project tree, including a live `.venv`. Not disposable. |
| `C:\Users\Gary\.conda\pkgs` | 5.45 | ZERO-YIELD TARGET, not a real 5.45 GB. `fsutil hardlink list` proves every large payload is a hard link into the anaconda install - e.g. `pkgs\cuda-nvrtc-12.1.105-0\lib\x64\nvrtc_static.lib` (306.6 MB) also resolves to `\ProgramData\anaconda3\Lib\x64\nvrtc_static.lib`; same for `cublasLt64_12.dll` (467.0 MB), `cufft64_11.dll` (181.5 MB), `cublas64_12.dll`, `curand64_10.dll` (MEASURED). Deleting the cache would free approximately 0 bytes while destroying conda's package cache. This also explains why the previous pass's `conda clean --all` reported nothing to reclaim, and means the headline "anaconda 21.1 GB" and this 5.45 GB partly double-count the same blocks. |
| `C:\Users\Gary\AppData\Local\Google` | 4.73 | FLAGGED as operator personal files. |
| `C:\Users\Gary\AppData\Local\Roblox` | 4.04 | FLAGGED as operator personal files. |
| `C:\Users\Gary\Desktop\jepa` | 2.45 | FLAGGED. Git repository and the live NeurIPS submission. |
| `C:\Users\Gary\.copilot\session-state` + `session-store.db` (+ `-wal`) | 1.76 | Copilot CLI session state: 101 checkpoints and 525 persistent artefacts, plus the session history database. This is the durable record; only the separate `logs\` tree was disposable. |
| `C:\Users\Gary\AppData\Local\Microsoft\Edge` | 1.59 | Browser cache/profile. The brief explicitly excludes browsers. |
| `C:\ProgramData\Tencent` (WeGame) | 0.97 | FLAGGED as a game launcher. |
| `C:\Users\Gary\AppData\Local\Microsoft\WinGet\Packages` | 0.60 | Live portable installs, not a cache. Two entries are on the user PATH: `GitHub.Copilot_...` and `GitHub.cli_...\bin` (MEASURED, PATH read). Deleting would break `gh` and `copilot`. |
| `C:\Users\Gary\AppData\Local\Discord` | 0.54 | FLAGGED as operator personal files. |
| `C:\Users\Gary\AppData\Local\Steam` | 0.50 | FLAGGED as operator personal files. |
| `C:\ProgramData\NVIDIA Corporation\Nsight` | 0.49 | Driver/toolkit adjacent. `C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\` is on the machine PATH (MEASURED). Safety rule: never touch a driver. |
| `C:\ProgramData\NVIDIA\NGX` | 0.47 | DLSS/NGX model payloads shipped with the display driver. Safety rule: never touch a driver. |
| `C:\Users\Gary\AppData\Local\Microsoft\Windows\Explorer` | 0.31 | Thumbnail and icon cache. Regenerates, but is held open by the live `explorer.exe`; deletion would mostly fail and the yield is marginal. Reported rather than forced. |

### Targets investigated and found already empty or absent (MEASURED)

`C:\$Recycle.Bin` 0 items / 0.000 GB; pip cache 0 bytes (`pip cache info`);
`C:\Windows\MEMORY.DMP` absent; `C:\Windows.old` absent; `C:\$WinREAgent` absent;
`C:\Windows\Minidump` 0.000 GB; npm, yarn, nuget, `.cargo`, `.gradle`, go module
cache, `uv` cache and torch hub cache all absent; `.cache\huggingface` 0.003 GB.
The brief's step 7 (per-user tool caches) therefore had no yield available.

---

## Remaining levers, in descending value (not executed)

1. `powercfg /hibernate off` - 12.77 GB, MEASURED size. Needs operator consent; also disables fast startup.
2. Relocate `C:\jepa_data\mirage_soft_guides` (44.7 GB) to D: - explicitly counter-indicated by `docs\experiments\masking\cover_random_campaign.md:116-117,329` and by D: being a spinning HDD. Only worth revisiting once the paused replication is finished.
3. Orphaned `C:\Windows\Installer` entries - up to a few GB, but needs a third-party analyser and operator sign-off.
4. `dism /startcomponentcleanup /resetbase` - up to 3.68 GB, permanently blocks uninstalling installed updates.

## Method notes

- Every removal step measured `Win32_LogicalDisk.FreeSpace` for C: immediately before and after; those deltas are the numbers in table A.
- Locked files were skipped silently and counted (20 in step 3, 0 elsewhere).
- Directory sizes are logical byte sums and can exceed reclaimable space where NTFS compression (Panther) or hard links (`.conda\pkgs`) are involved; both cases are called out above.
- Claims are labelled MEASURED where produced by a command run in this session, and INFERRED where reasoned from that evidence.
