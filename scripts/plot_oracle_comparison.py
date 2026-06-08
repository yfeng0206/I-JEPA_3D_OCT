"""Oracle vs random masking — comparison figures (frozen sweep + fine-tune).

Final committed numbers (see docs/experiments/frozen/oracle_meanpool_sweep.md and
docs/experiments/finetune/oracle_finetune.md). Significance from the paired bootstrap
(scripts/bootstrap_frozen_meanpool.py, scripts/bootstrap_finetune.py, B=2000).

Outputs to results/summary/: oracle_frozen_curve.png, oracle_finetune_bars.png, oracle_summary.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'summary')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'legend.fontsize': 11, 'axes.grid': True, 'grid.alpha': 0.3})
C_ORACLE, C_RANDOM = '#d1495b', '#3a6ea5'

# Frozen MeanPool sweep (Test AUC)
EP = [50, 75, 100]
ORACLE_F = [0.8740, 0.8836, 0.8855]
RANDOM_F = [0.8641, 0.8723, 0.8746]

# Fine-tune, 3 probes (Test AUC) + paired-bootstrap significance
PROBES = ['MeanPool', 'CrossAttnPool', 'd=1']
ORACLE_FT = [0.8947, 0.8937, 0.8901]
RANDOM_FT = [0.8868, 0.8872, 0.8878]
FT_STARS = ['***', '**', 'ns']  # p = 0.001, 0.009, 0.26


def frozen_panel(ax):
    ax.plot(EP, ORACLE_F, 'o-', color=C_ORACLE, label='Oracle masking', markersize=7)
    ax.plot(EP, RANDOM_F, 's--', color=C_RANDOM, label='Random masking', markersize=7)
    for x, yo, yr in zip(EP, ORACLE_F, RANDOM_F):
        ax.annotate(f'+{yo - yr:.3f}', (x, (yo + yr) / 2), fontsize=9, ha='center', color='0.3')
    ax.set_xlabel('Pretraining epoch')
    ax.set_ylabel('Test AUC')
    ax.set_xticks(EP)
    ax.set_ylim(0.855, 0.892)
    ax.set_title('Frozen MeanPool probe\npaired bootstrap p<0.0005 at every epoch')
    ax.legend(loc='lower right')


def ft_panel(ax):
    x = np.arange(len(PROBES))
    w = 0.38
    ax.bar(x - w / 2, RANDOM_FT, w, color=C_RANDOM, label='Random masking')
    ax.bar(x + w / 2, ORACLE_FT, w, color=C_ORACLE, label='Oracle masking')
    for i, (r, o, s) in enumerate(zip(RANDOM_FT, ORACLE_FT, FT_STARS)):
        ax.text(i + w / 2, o + 0.0007, s, ha='center', fontsize=11, fontweight='bold')
        ax.text(i - w / 2, 0.8615, f'{r:.3f}', ha='center', va='bottom', fontsize=8, color='white', rotation=90)
        ax.text(i + w / 2, 0.8615, f'{o:.3f}', ha='center', va='bottom', fontsize=8, color='white', rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(PROBES)
    ax.set_ylim(0.86, 0.906)
    ax.set_ylabel('Test AUC')
    ax.set_title('Fine-tune (oracle ep100 vs random ep100)\n*** p<0.001   ** p<0.01   ns not sig.')
    ax.legend(loc='upper right', framealpha=0.95)


for name, fn in [('oracle_frozen_curve', frozen_panel), ('oracle_finetune_bars', ft_panel)]:
    fig, ax = plt.subplots(figsize=(7, 5))
    fn(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=150)
    plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
frozen_panel(a1)
ft_panel(a2)
fig.suptitle('Anatomy-guided (oracle) vs random masking — FairVision glaucoma, 3000-volume Test', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'oracle_summary.png'), dpi=150)
plt.close(fig)
print('saved: oracle_frozen_curve.png, oracle_finetune_bars.png, oracle_summary.png')
