"""Regenerate fig_pareto_frontier.png from 3,257 aggregation.

Visibility fixes vs. v1:
  * Short defense IDs (D1..D12) on points -> legend on the right maps to full names.
  * Removes literal '\_' artefact (matplotlib wasn't in mathmode).
  * Small dx/dy nudges per point for the dense (S_P1~0.4, y~0.65-0.70) cluster.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'code/results/aggregated_summary.json') as f:
    agg = json.load(f)

cd = agg['cross_dataset_per_defense']

# Mapping defense_key -> short ID (paper notation)
SHORT = {
    'd1_raw': 'D1', 'd2_gaussian_dp': 'D2', 'd3_laplace_dp': 'D3',
    'd4_dp_ae': 'D4', 'd5_affine': 'D5', 'd6_per_seg_affine': 'D6',
    'd7_ope': 'D7', 'd8_doppelganger': 'D8', 'd9_kanon': 'D9',
    'd10_fe_ip': 'D10', 'd11_vfae': 'D11', 'd12_ib': 'D12',
}
LEGEND = {
    'D1': 'raw (no defense)', 'D2': 'Gaussian-DP', 'D3': 'Laplace-DP',
    'D4': 'DP-AE', 'D5': 'global affine', 'D6': 'per-segment affine',
    'D7': 'OPE', 'D8': 'Doppelganger', 'D9': 'k-anonymity',
    'D10': 'FE-IP', 'D11': 'VFAE-TS', 'D12': 'IB',
}

# Manual label-anchor table to break overlaps in the dense
# (S_P1 in [0.29, 0.47], y in [0.65, 0.70]) cluster.
# value = (ha, va, dx, dy)
ANCHOR = {
    'd1_raw':           ('right',  'center', -0.018,  0.00),
    'd2_gaussian_dp':   ('right',  'top',    -0.020, -0.018),
    'd3_laplace_dp':    ('left',   'top',     0.020, -0.018),
    'd4_dp_ae':         ('right',  'center', -0.020,  0.00),
    'd5_affine':        ('left',   'center',  0.020,  0.00),
    'd6_per_seg_affine':('right',  'bottom', -0.020,  0.020),
    'd7_ope':           ('left',   'center',  0.020,  0.00),
    'd8_doppelganger':  ('center', 'bottom',  0.000,  0.025),
    'd9_kanon':         ('right',  'center', -0.020,  0.00),
    'd10_fe_ip':        ('left',   'center',  0.020,  0.00),
    'd11_vfae':         ('right',  'top',    -0.018, -0.018),
    'd12_ib':           ('left',   'top',     0.020, -0.018),
}

points = []
for de, m in cd.items():
    x = m['S_P1']
    y = min(m['S_P3-CI'], m['S_P4'])
    points.append((de, x, y, m['HM']))

fig, ax = plt.subplots(figsize=(9.0, 6.5))

# Impossibility region
ax.add_patch(plt.Rectangle((0.8, 0.8), 0.2, 0.2, alpha=0.15, color='red',
                            label='Empirical impossibility region'))
ax.text(0.83, 0.92, 'No defense\nlies here', color='darkred',
        fontsize=10, ha='left', va='center')

# Distinct colors per point
colors = plt.cm.tab20(np.linspace(0, 1, len(points)))
for (de, x, y, hm), c in zip(points, colors):
    ax.scatter(x, y, s=210, c=[c], edgecolor='black',
               linewidth=1.0, zorder=4)
    ha, va, dx, dy = ANCHOR.get(de, ('left', 'center', 0.012, 0.0))
    ax.annotate(SHORT[de], (x + dx, y + dy), fontsize=10,
                fontweight='bold', ha=ha, va=va, zorder=5)

# Pareto frontier: skyline of upper-right points
sorted_pts = sorted(points, key=lambda p: -p[1])
pareto = []
max_y = -1.0
for de, x, y, hm in sorted_pts:
    if y > max_y:
        pareto.append((x, y))
        max_y = y
pareto.sort(key=lambda p: p[0])
if len(pareto) >= 2:
    px, py = zip(*pareto)
    ax.plot(px, py, color='#444444', linewidth=1.6, linestyle='-',
            alpha=0.7, zorder=2, label='Empirical Pareto frontier')

ax.axhline(0.8, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(0.8, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlim(-0.06, 1.10)
ax.set_ylim(0.55, 1.04)  # zoom — all points are above 0.6
ax.set_xlabel(r'$S_{P_1}$ (algebraic equivariance / utility)', fontsize=12)
ax.set_ylabel(r'$\min(S_{P_3\text{-CI}},\ S_{P_4})$ (worst-case privacy)',
              fontsize=12)
ax.set_title(f'Quadrilemma Pareto Frontier (cross-dataset; n={agg["n_results"]} cells)',
             fontsize=12)
ax.grid(True, alpha=0.3)

# Two-column legend on the right: defense ID -> name
legend_lines = [f'{k} = {v}' for k, v in LEGEND.items()]
text = '\n'.join(legend_lines)
ax.text(1.02, 0.99, text, transform=ax.transAxes, fontsize=8.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.95))

ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
out_path = ROOT / 'figures/fig_pareto_frontier.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'[OK] Saved: {out_path}')

# Also overwrite the canonical artifact location for downstream consistency.
alt = ROOT / 'code/results/paper_artifacts/figures/fig_pareto_frontier.png'
alt.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(alt, dpi=200, bbox_inches='tight')
print(f'[OK] Saved: {alt}')
plt.close()
