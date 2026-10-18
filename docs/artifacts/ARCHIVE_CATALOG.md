# Research artifact catalog

Archive snapshot: **2026-10-17 PDT**. This index labels existing research assets
and ongoing backups; it does not add experiment results or change the registered
analysis. Sources remain in their original locations.

## Paper, figures, results and methods

| Artifact | Canonical source |
|---|---|
| Workshop paper and poster guide | [Paper README](../../paper/genai4health2026/README.md) |
| Registered continuation campaign | [cr_seed_v1 protocol](../../paper/genai4health2026/PREREGISTRATION_cr_seed_v1.md) |
| Paper source | [main_submission.tex](../../paper/genai4health2026/main_submission.tex) |
| Print poster draft, 61 × 85 cm | [poster_print.pdf](../../poster/genai4health2026/poster_print.pdf) |
| Reviewed paper illustrations | [Figures](../../paper/genai4health2026/figures/), [producer and provenance registry](../../autopilot/release_assets.json) |
| Historical downstream weights/prediction mapping | [ARTIFACT_MAP.md](../../results/downstream/ARTIFACT_MAP.md), [machine-readable map](../../results/downstream/ARTIFACT_MAP.json) |
| Historical results | [Results](../../results/), [experiment documentation](../experiments/README.md) |
| MIRAGE adapter design | [Architecture](../architecture/mirage_adapter.md) |
| MIRAGE guidance and historical training | [Guided masking](../experiments/mirage_guided_masking.md), [training notes](../experiments/pretraining/mirage_100ep.md) |
| Historical ENVELOPE / anatomy configurations | [ENVELOPE](../../configs/patch_mirage_envelope.yaml), [anatomy](../../configs/patch_mirage_anatomy.yaml) |

The historical artifact map predates the camera-ready campaign and does not
enumerate every new seed. Figure files and their evidence receipts should remain
together; an output alone is not a complete reproducibility record.

## Names and evaluation caveats

- Historical **`oracle` means CENTROID**, the intensity-guided policy. Retain
  existing `oracle-anatomical-100ep` paths for compatibility; the alias does not
  imply ground-truth segmentation guidance.
- Historical ANATOMY-V2 checkpoints span multiple runs. Its cited epoch-50
  checkpoint comes from the original anatomy continuation, while clean fp32
  epoch-75 comes from `blob_fp32_ep56`. The precision-spliced
  `blob_resume_ep56` weights are a different artifact and must not be substituted.
- Historical COVER describes the original implementation, including its
  documented delivered-mask limitations. Corrected-code diagnostics are not
  newly trained corrected-model results.
- `cr_seed_v1` models are continuations from a **shared epoch-25 ancestor**,
  rather than independent pretraining from scratch. An archived checkpoint is
  not evidence of an on-time completed probe or a performance conclusion.

## Private Hugging Face checkpoint backup

Weights are archived in the existing **private** repository
[`yfeng0206/ijepa-3d-oct-checkpoints`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints).
Access requires authorization. Existing historical folders and filenames are
retained; new campaign weights use `camera-ready/cr_seed_v1/`.

The local backup ledger at this snapshot records the following completed
epoch-50 pins as remotely verified by **SHA-256 and byte count**. Commit links
identify the recorded upload revisions; local copies remain retained.

| Run | Policy / seed | Recorded verified upload revision |
|---|---|---|
| R1 | RANDOM / 1234 | [`f9212704`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/tree/f9212704729f6d27b166616c74372f1d3e65a285/camera-ready/cr_seed_v1/cr_seed_v1_random_s1234) |
| C1 | CENTROID / 1234 | [`cb567f34`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/tree/cb567f34d5ecb5d0bc19c7dbdefeff9394629ef3/camera-ready/cr_seed_v1/cr_seed_v1_centroid_s1234) |
| E1 | ENVELOPE / 1234 | [`63ead98c`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/tree/63ead98cbbcc9ea863d1fd565b7c2ecd7b75be55/camera-ready/cr_seed_v1/cr_seed_v1_envelope_s1234) |
| R2 | RANDOM / 5678 | [`4b44ec3d`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/tree/4b44ec3d1085d2cf820de70dc342391d67d4c8e7/camera-ready/cr_seed_v1/cr_seed_v1_random_s5678) |
| C2 | CENTROID / 5678 | [`151bdebc`](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/tree/151bdebc46c0727cacd360c67e1ef2d0ea6f3fe5/camera-ready/cr_seed_v1/cr_seed_v1_centroid_s5678) |

These five pins total **7,537,669,316 bytes (7.02 GiB)**. They are a subset,
not a claim that all campaign artifacts are backed up. No sealed campaign test metrics or per-case data are included here.
Filename existence alone is insufficient verification of an online backup.

## Selected MIRAGE archive

The selected historical MIRAGE plan contains **10 checkpoints totaling
10,489,620,950 bytes (9.77 GiB)**, plus 44 small provenance candidates. Original
folder labels are retained: `mergedv3`, `mergedv3-512`, `mergedv3-base-512`, and
`official-vitlarge`. The archived source checkout reports revision
`930523b209dad3dd2eadc7296fb7ce3f3eaa9924`, with local changes also preserved.
This snapshot does not independently certify the exact source that produced
every historical checkpoint. A filename containing `best` remains a filename
label, not a newly verified quality assessment.

The ten weights and **44 source/configuration/provenance files** are remotely
content-verified in `research-archive/mirage/`; the final
[archive manifest](https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints/blob/main/research-archive/mirage/ARCHIVE_MANIFEST.json)
is verified too. Large files use SHA-256/LFS identifiers and byte counts; small
Git blobs are checked against their content-addressed object identifiers.

| Original experiment folder | Preserved filenames |
|---|---|
| `mergedv3` | `checkpoint-34.pth`, `checkpoint-best.pth`, `checkpoint-199.pth` |
| `mergedv3-512` | `checkpoint-34.pth`, `checkpoint-best.pth`, `checkpoint-199.pth` |
| `mergedv3-base-512` | `checkpoint-34.pth`, `checkpoint-best.pth` |
| `official-vitlarge/GOALS` | `checkpoint-best.pth`, `checkpoint-best-trainable-state.pth` |

Numbered MIRAGE filenames are retained as checkpoint indices, without inferring
an epoch convention. Source, configuration and aggregate-log provenance
accompany the selected weights; clinical images,
per-case arrays and layer/matrix dumps are excluded. This archival selection
does not establish segmentation accuracy or clinical validation.

## Third-party downloads and restoration

These are upstream pretrained models, separate from user-trained JEPA/MIRAGE
artifacts. Preserve their licenses, configuration, processors/tokenizers and any
custom classifiers. Restore from the original publisher at a pinned revision;
do not redistribute gated weights through this project.

```powershell
hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --local-dir ./restored/qwen3_vl --include 'model-*.safetensors'
hf download allenai/MolmoPoint-8B --revision 188130f961c8e0888a34e11121a1423c461a01ba --local-dir ./restored/molmo --include 'model-*.safetensors'
hf download facebook/dinov3-vit7b16-pretrain-lvd1689m --revision b80367753773648a6793235ab9c65cdbb029506f --local-dir ./restored/dinov3 --include 'model-*.safetensors'
```

DINO requires upstream access/license acceptance and an authenticated account.
These commands restore weight shards; they are not a complete configuration
installation. Verify downloads against the retained private SHA-256 receipts.
Upstream ETag/size agreement alone does not prove current local bytes match.
Only the first two Qwen shards had completed local streamed SHA-256 verification
when the restore notes were prepared; the remaining upstream shards had metadata
verification only.

## Local retention and cleanup status

**No local deletion occurred in this cleanup pass: automatic approval review
blocked deletion.** Upload verification does not authorize inference that local
files have been removed or space reclaimed.

All campaign pins, feature caches, shared ancestors, guides, datasets, stop/seal
records and active/queued runtime dependencies remain retained. Keep clinical
data, patient identifiers, absolute personal paths, credentials and sealed test
metrics out of public archive indexes. Final camera-ready analysis continues to
follow the registered freeze and release workflow.
