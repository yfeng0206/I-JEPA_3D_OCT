# Cleanup accounting, 2026-08-26

Everything removed, and everything deliberately kept. Written because several
things that *looked* disposable turned out to be load-bearing, and the reasoning
matters more than the totals.

Detailed reports: `DISK_CLEANUP_FULL.md`, `CLEANUP_DECISIONS.md`, `GITHUB_AUDIT.md`.

---

## Totals

| where | reclaimed |
|---|---|
| C: (SSD) | **12.34 GB** - free went 17.23 GB (7.45%) to **29.56 GB (12.78%)** |
| D: (HDD) | **6.28 GB** |
| repo working tree | **29.7 MB** |
| repo, archived not deleted | 150 files, 11.94 MB, reversible via `git mv` |

---

## A. REMOVED

### C: drive
| item | GB | why it was safe |
|---|---|---|
| Copilot CLI diagnostic logs | 5.96 | Diagnostics only. Session-state was preserved. |
| WinSxS component store | 5.01 | Two DISM passes, **no `/ResetBase`**, so existing updates stay uninstallable. |
| Temp directories | 0.47 | Regenerable by definition. |
| Delivery Optimization cache | 0.45 | Peer-to-peer update cache; Windows refetches. |
| Panther upgrade logs | 0.38 | Windows setup logs from a completed upgrade. |
| WER, crash dumps, rotated CBS logs | 0.06 | Post-mortem artifacts, no live consumer. |
| Windows Update download cache | 0.003 | Refetched on demand. |

### D: drive
| item | GB | why it was safe |
|---|---|---|
| `checkpoints_hf/random-posfix-100ep` | 5.62 | A download of the Hub copy. SHA-256 verified **byte-identical** to the local ancestor before removal. |
| `tmp4dbjtgj3`, `tmp_guide_mixed`, `torch_cache`, `matplotlib-cache`, `cache` | 0.66 | Scratch and regenerable caches. |
| Empty run dir `frozen_meanpool_bridge_ep41` | 0 | Contained no files at all - an abandoned probe. |

### Repository
| item | size | why it was safe |
|---|---|---|
| 5 superseded Overleaf ZIPs | 29.7 MB | Untracked build outputs. The validated ZIP ships to Downloads. |
| `__pycache__` (15 dirs), `.pytest_cache` | 3.2 MB | Regenerable. |
| `logs/patch_run3_log.csv` | 1.99 MB | Proven **byte-identical duplicate** (hash `7EA35BEC...`) of a surviving twin, left by a 2026-03-30 reorg. |

---

## B. RETAINED DESPITE BEING LARGE - flagged important

This is the part worth reading. Each of these looked like a cleanup candidate and
is not.

| item | size | why it must stay |
|---|---|---|
| **`C:\jepa_data`** | **70.8 GB** | `src/train_patch.py:349-353` **raises** if `slice_cache` is missing, and ten configs hard-code the path. It sits on the SSD **deliberately** - `docs/experiments/masking/cover_random_campaign.md:116-117,329` records both trees as "SSD, shared with blob" for dataloader throughput. **C: is the SSD; D: is a spinning HDD**, so relocating it to free 70 GB would slow every future training run. A paused replication needs it. |
| **`C:\ProgramData\anaconda3`** | **21.1 GB** | I assumed this was dead. **It is not.** Navigator `.pyc` imports on 2026-07-30 17:56 with a matching `PYTHONW.EXE` prefetch at 17:57, plus 1,868 `.pyc` reads on 2026-05-12. Left intact. |
| `C:\Users\Gary\.conda\pkgs` | 5.45 GB | **Hard-linked into anaconda3.** Deleting it frees zero bytes and would corrupt the environment. |
| **`D:\jepa_phase0\runs`** | **207 GB** | An automated "unreferenced" scan flagged 118 GB of this as deletable. **That scan was wrong.** Those dirs hold the trained models behind every arm - `patch_mirage_envelope` alone has 17 checkpoints including the ep100 the paper reports - plus `rep_random_s1234`, the paused replication's resume point. They scored zero because the evidence files cite checkpoint *paths*, not run-dir names. |
| `D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep` | 4.21 GB | Searched both drives: this is the **only local copy** of the headline CENTROID arm's ep50/75/100 weights. |
| `results/masking/structural_loss/regen01.json` | small | It **is** a byte-identical duplicate, and that identity **is** the reproducibility finding the paper reports. Deleting it would destroy the evidence. |
| `figS5_mask_statistics.png` | small | Labelled scratch by an earlier audit, but still `\includegraphics`-d at `main_submission.tex:796`. Deleting it breaks the build. |
| `scripts/run_guarded_probe.py`, `scripts/download_weights.py` | small | Labelled scratch, but both are invoked by the replication chain. |
| Two `.bak` files under `paper/` | small | Match **no revision in git history**, so a `git rm` would destroy unique content permanently. |
| `hiberfil.sys` | 12.77 GB | Available via `powercfg /hibernate off`, which would reach about 18.3% free. **Needs operator consent** - it disables hibernate and fast startup. Not run. |
| Games and launchers | large | Steam, GOG, Riot, Xbox, Battle.net, WeGame, and browser/Roblox/Discord caches. Personal files, not project data. |

---

## C. Found but NOT fixed - needs an operator decision

1. **The double-blind submission is on a public repository under the author's real
   name.** `main_submission.pdf` is world-readable at `yfeng0206/I-JEPA_3D_OCT`
   while the paper states "links withheld for anonymity". The manuscript passes
   its own anonymity gate; the exposure is the repository. Changing visibility
   affects a collaborator's clone, so it was left alone. See `GITHUB_AUDIT.md`.

2. **272.8 MB of history bloat** from 20 tracked `paper/dist/*.zip` blobs, caused
   by a `!dist/*.zip` negation in `paper/.gitignore` that overrode the root
   ignore for months. The negation is now removed so no more can accumulate, but
   clearing what exists needs a history rewrite, which breaks the teammate's
   clone and is forbidden here.

---

## The pattern worth remembering

Four separate times today an automated or remembered label said "safe to delete"
and direct verification said otherwise: the run directories, anaconda, the
`checkpoints_hf` oracle copy, and the three mislabelled scratch files. In every
case the label was cheap and the check was cheap too. The check is what mattered.
