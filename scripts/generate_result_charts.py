"""Generate result charts for README from final notebook results."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / 'outputs'
OUT.mkdir(exist_ok=True)

STYLE = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}
plt.rcParams.update(STYLE)

FOLD_AUCS = {
    'LightGBM':  [0.93526, 0.94033, 0.95148, 0.93951, 0.93051],
    'XGBoost':   [0.93366, 0.93972, 0.95051, 0.93875, 0.92962],
    'CatBoost':  [0.93234, 0.93687, 0.95046, 0.93805, 0.92889],
    'LGB DART':  [0.92724, 0.93631, 0.94606, 0.93492, 0.92748],
}

ENSEMBLE_AUCS = {
    'LightGBM':            0.94004,
    'XGBoost':             0.93922,
    'CatBoost':            0.93793,
    'LGB DART':            0.93506,
    'Rank 3-model':        0.93995,
    'Rank 4-model (+DART)':0.93943,
    'Weighted 4-model':    0.94019,
    'Stacked (LR meta)':   0.93848,
}

COLORS = {
    'LightGBM':  '#2196F3',
    'XGBoost':   '#FF9800',
    'CatBoost':  '#4CAF50',
    'LGB DART':  '#9C27B0',
}
ENSEMBLE_COLOR = '#F44336'


# ── Chart 1: Per-fold AUC ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

folds = [1, 2, 3, 4, 5]
x = np.arange(len(folds))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for (model, aucs), offset in zip(FOLD_AUCS.items(), offsets):
    bars = ax.bar(x + offset * width, aucs, width, label=model,
                  color=COLORS[model], alpha=0.85, edgecolor='white')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Per-Fold AUC — GroupKFold(5) by Race×Year', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.set_ylim(0.915, 0.960)
ax.legend(loc='lower right', fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

plt.tight_layout()
plt.savefig(OUT / 'fold_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fold_auc_comparison.png')


# ── Chart 2: Ensemble OOF AUC comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

names = list(ENSEMBLE_AUCS.keys())
aucs  = list(ENSEMBLE_AUCS.values())

base_models = {'LightGBM', 'XGBoost', 'CatBoost', 'LGB DART'}
bar_colors = [
    COLORS.get(n, ENSEMBLE_COLOR) for n in names
]
# Highlight best
best_idx = aucs.index(max(aucs))
bar_colors[best_idx] = '#E53935'

bars = ax.barh(names, aucs, color=bar_colors, alpha=0.88, edgecolor='white', height=0.6)

for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.0002, bar.get_y() + bar.get_height() / 2,
            f'{auc:.5f}', va='center', ha='left', fontsize=9.5)

ax.set_xlabel('OOF AUC-ROC', fontsize=12)
ax.set_title('OOF AUC — Model & Ensemble Comparison', fontsize=13, fontweight='bold')
ax.set_xlim(0.930, 0.946)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

legend_patches = [
    mpatches.Patch(color=COLORS['LightGBM'],  label='LightGBM'),
    mpatches.Patch(color=COLORS['XGBoost'],   label='XGBoost'),
    mpatches.Patch(color=COLORS['CatBoost'],  label='CatBoost'),
    mpatches.Patch(color=COLORS['LGB DART'],  label='LGB DART'),
    mpatches.Patch(color=ENSEMBLE_COLOR,       label='Ensemble (best)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved ensemble_comparison.png')


# ── Chart 3: Mean CV AUC + std ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

models  = list(FOLD_AUCS.keys())
means   = [np.mean(v) for v in FOLD_AUCS.values()]
stds    = [np.std(v)  for v in FOLD_AUCS.values()]
colors  = [COLORS[m] for m in models]

ax.bar(models, means, yerr=stds, color=colors, alpha=0.85,
       capsize=6, edgecolor='white', error_kw={'linewidth': 1.8})

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.0008, f'{m:.5f}', ha='center', fontsize=9.5)

ax.set_ylabel('Mean OOF AUC-ROC', fontsize=12)
ax.set_title('Mean CV AUC ± Std Dev (5-Fold)', fontsize=13, fontweight='bold')
ax.set_ylim(0.925, 0.950)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

plt.tight_layout()
plt.savefig(OUT / 'mean_cv_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved mean_cv_auc.png')

print('\nAll charts saved to outputs/')
