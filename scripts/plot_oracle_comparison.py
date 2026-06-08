"""Oracle vs random masking — comparison figures (frozen sweep + fine-tune).

Final committed numbers (see docs/experiments/frozen/oracle_meanpool_sweep.md and
docs/experiments/finetune/oracle_finetune.md). Significance from the paired bootstrap
(scripts/bootstrap_frozen_meanpool.py, scripts/bootstrap_finetune.py, B=2000).

Horizontal grouped bars in the style of results/summary/probe_ranking_ep100.png.
Outputs to results/summary/: oracle_frozen_bars.png, oracle_finetune_bars.png, oracle_summary.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'summary')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 11})
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


def grouped_barh(ax, cats, random_vals, oracle_vals, xlim, title, stars=None, legend=True):
    y = np.arange(len(cats))
    h = 0.38
    ax.barh(y + h / 2, oracle_vals, h, color=C_ORACLE, label='Oracle masking')
    ax.barh(y - h / 2, random_vals, h, color=C_RANDOM, label='Random masking')
    for i in range(len(cats)):
        lbl = f'{oracle_vals[i]:.4f}' + (f'  {stars[i]}' if stars else '')
        ax.text(oracle_vals[i] + 0.0004, y[i] + h / 2, lbl, va='center', fontsize=9, fontweight='bold')
        ax.text(random_vals[i] + 0.0004, y[i] - h / 2, f'{random_vals[i]:.4f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlim(*xlim)
    ax.set_xlabel('Test AUC')
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.3)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)


def frozen(ax, title='Frozen MeanPool — oracle vs random\npaired bootstrap p<0.0005 at every epoch', legend=True):
    grouped_barh(ax, [f'ep{e}' for e in EP], RANDOM_F, ORACLE_F, (0.855, 0.895), title, legend=legend)


def finetune(ax, title='Fine-tune — oracle vs random\n*** p<0.001   ** p<0.01   ns not sig.', legend=True):
    grouped_barh(ax, PROBES, RANDOM_FT, ORACLE_FT, (0.86, 0.905), title, stars=FT_STARS, legend=legend)


fig, ax = plt.subplots(figsize=(7.5, 4.8)); frozen(ax)
fig.savefig(os.path.join(OUT, 'oracle_frozen_bars.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.8)); finetune(ax)
fig.savefig(os.path.join(OUT, 'oracle_finetune_bars.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
frozen(a1, 'Frozen MeanPool sweep\npaired bootstrap p<0.0005 at every epoch', legend=False)
finetune(a2, 'Fine-tune, 3 probes\n*** p<0.001   ** p<0.01   ns not sig.', legend=False)
handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
fig.suptitle('Anatomy-guided (oracle) vs random masking — FairVision glaucoma, 3000-volume Test', fontsize=14, y=0.99)
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.28)
fig.savefig(os.path.join(OUT, 'oracle_summary.png'), dpi=150); plt.close(fig)
print('saved: oracle_frozen_bars.png, oracle_finetune_bars.png, oracle_summary.png')
