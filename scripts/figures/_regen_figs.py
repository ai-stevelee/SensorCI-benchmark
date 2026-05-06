"""Regenerate fig_radar_per_defense.png from 3,257 aggregation.

Visibility fixes vs. v1:
  * Cleaner axis labels (mathmode + larger font; tick labels removed because
    they collided with axis labels at small panel size).
  * Bold subplot titles in two lines: ID + short name | metrics.
  * Reference circle at 0.8 stays red dashed.
  * Tighter padding so the 3x4 grid uses page width without dead space.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'code/results/aggregated_summary.json') as f:
    agg = json.load(f)

cd = agg['cross_dataset_per_defense']
ordered = sorted(cd.items(), key=lambda kv: -kv[1]['HM'])

SHORT = {
    'd1_raw': 'D1: raw', 'd2_gaussian_dp': 'D2: Gaussian-DP',
    'd3_laplace_dp': 'D3: Laplace-DP', 'd4_dp_ae': 'D4: DP-AE',
    'd5_affine': 'D5: affine', 'd6_per_seg_affine': 'D6: per-seg affine',
    'd7_ope': 'D7: OPE', 'd8_doppelganger': 'D8: Doppelganger',
    'd9_kanon': 'D9: k-anon', 'd10_fe_ip': 'D10: FE-IP',
    'd11_vfae': 'D11: VFAE-TS', 'd12_ib': 'D12: IB',
}

n_defenses = len(ordered)
ncols = 4
nrows = math.ceil(n_defenses / ncols)

axes_labels = [r'$S_{P_1}$', r'$S_{P_2}$', r'$S_{P_3\text{-CI}}$', r'$S_{P_4}$']
n_axes = len(axes_labels)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles_closed = angles + [angles[0]]

fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.2 * nrows),
                         subplot_kw=dict(polar=True))
fig.suptitle(f'Per-defense Quadrilemma scores '
             f'(cross-dataset; n={agg["n_results"]} cells; ranked by HM descending)',
             fontsize=14, y=1.01)

for idx, (defense, m) in enumerate(ordered):
    ax = axes[idx // ncols, idx % ncols]
    values = [m['S_P1'], m['S_P2'], m['S_P3-CI'], m['S_P4']]
    values_closed = values + [values[0]]
    ax.plot(angles_closed, values_closed, color='#1f77b4', linewidth=2.2)
    ax.fill(angles_closed, values_closed, color='#1f77b4', alpha=0.28)
    ref = [0.8] * (n_axes + 1)
    ax.plot(angles_closed, ref, color='red', linewidth=1.0,
            linestyle='--', alpha=0.6, label='target = 0.8')

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=11)
    # Hide y-tick labels (numbers) — they were colliding with axis labels;
    # keep only the gridline at 0.5 and 1.0 for visual scale.
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', pad=6)

    title = f"{SHORT[defense]}\nHM={m['HM']:.2f}  WA={m['WA']:.2f}"
    ax.set_title(title, fontsize=11, pad=18, fontweight='bold')
    ax.grid(True, alpha=0.35)

# Hide unused panels
for idx in range(n_defenses, nrows * ncols):
    fig.delaxes(axes[idx // ncols, idx % ncols])

plt.tight_layout()
out_path = ROOT / 'figures/fig_radar_per_defense.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'[OK] Saved: {out_path}')

alt = ROOT / 'code/results/paper_artifacts/figures/fig_radar_per_defense.png'
alt.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(alt, dpi=200, bbox_inches='tight')
print(f'[OK] Saved: {alt}')
plt.close()
