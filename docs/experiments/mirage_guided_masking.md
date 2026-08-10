# MIRAGE-guided masking (disambiguation)

The phrase **MIRAGE-guided masking** is ambiguous in this repository. It refers to two different masking modes, and the difference is the shape of the I-JEPA target, not whether MIRAGE is used.

| Mode | What MIRAGE controls | Target shape | Role | Config | Measured targeting |
|---|---|---|---|---|---|
| `mirage_envelope` | Placement on the retina | Ordinary rectangles | Baseline | `configs/patch_mirage_envelope.yaml` | 30.7% of masked cells on anatomy; 3.57% dead targets |
| `mirage_anatomy` | The tissue-shaped target itself | Connected irregular anatomy blobs | Contribution | `configs/patch_mirage_anatomy.yaml` | 72.1% of masked cells on anatomy; 2.05% dead targets |

MIRAGE guidance alone is not the novelty. The rectangle baseline already uses MIRAGE for placement. The method claim is anatomy-shaped masking: target blocks are shaped to tissue instead of remaining rectangles.

For current masking documentation, start with [`masking/`](masking/). For legacy completed pretraining and frozen-probe records, see [`pretraining/`](pretraining/) and [`frozen/`](frozen/).
