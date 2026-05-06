"""Fit C_1, C_2 per dataset using least-squares regression on Theorem 2 trade-off.

Theorem 2: alpha_1 + epsilon_2 + min_r epsilon_{3,r} >= C_1 * tau_4 - C_2

Operationalize:
  LHS_i = (1 - S_P1_i) + (1 - S_P2_i) + (1 - S_P3-CI_i) = 3 - (S_P1_i + S_P2_i + S_P3-CI_i)
  x_i = S_P4_i (proxy for tau_4)

We fit a linear regression  LHS_i = slope * x_i + intercept + residual_i.
Identifying slope = C_1 and -intercept = C_2 gives the empirical trade-off
relationship (in expectation, not strict envelope). Both forms are reported:
  - regression slope / intercept (empirical trend)
  - lower-envelope (C_1, C_2) such that LHS >= C_1 * x - C_2 for all observed cells
    (held empirically, not provably).
"""

import json
from collections import defaultdict
import numpy as np

with open('code/results/aggregated_summary.json') as f:
    agg = json.load(f)

per = agg['per_defense_dataset']
by_ds = defaultdict(list)
for key, m in per.items():
    by_ds[m['dataset']].append(m)


def fit_regression(points):
    """OLS regression of y = slope*x + intercept. Returns (slope, intercept, R^2)."""
    pts = np.array(points)
    xs, ys = pts[:, 0], pts[:, 1]
    if len(pts) < 2 or np.std(xs) < 1e-12:
        return 0.0, float(np.mean(ys)), 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def fit_lower_envelope(points, slope_hint):
    """Given a target slope, find the tight lower envelope intercept.

    Lower envelope: y >= slope*x - C2  for all observed (x, y).
    Tight C2 = max_i (slope*x_i - y_i).
    """
    pts = np.array(points)
    xs, ys = pts[:, 0], pts[:, 1]
    c2 = float(np.max(slope_hint * xs - ys))
    return slope_hint, c2


print('Empirical trade-off fits per dataset (regression + lower envelope):')
print(f'{"dataset":>10} | {"slope C1":>9} | {"-int C2":>8} | {"R^2":>5} | '
      f'{"env C2":>7}')
print('-' * 60)
results = {}
all_pts = []
for ds, ms in sorted(by_ds.items()):
    pts = []
    for m in ms:
        x = m['S_P4']
        y = 3 - (m['S_P1'] + m['S_P2'] + m['S_P3-CI'])
        pts.append((x, y))
        all_pts.append((x, y))
    slope, intercept, r2 = fit_regression(pts)
    c1_reg = slope
    c2_reg = -intercept
    _, c2_env = fit_lower_envelope(pts, slope_hint=c1_reg)
    results[ds] = {
        'C_1_regression': c1_reg,
        'C_2_regression': c2_reg,
        'R_squared': r2,
        'C_2_lower_envelope_at_same_slope': c2_env,
    }
    print(f'{ds:>10} | {c1_reg:>9.3f} | {c2_reg:>8.3f} | {r2:>5.2f} | '
          f'{c2_env:>7.3f}')

print()
print('Pooled (60 cells across all 5 datasets):')
slope, intercept, r2 = fit_regression(all_pts)
_, c2_env = fit_lower_envelope(all_pts, slope_hint=slope)
print(f'   slope (C_1) = {slope:.3f}, -intercept (C_2) = {-intercept:.3f}, '
      f'R^2 = {r2:.2f}, env C_2 = {c2_env:.3f}')

print()
c1_vals = [r['C_1_regression'] for r in results.values()]
c2_vals = [r['C_2_regression'] for r in results.values()]
print(f'Per-dataset C_1 range: [{min(c1_vals):.3f}, {max(c1_vals):.3f}]')
print(f'Per-dataset C_2 range: [{min(c2_vals):.3f}, {max(c2_vals):.3f}]')

out = {
    'method': ('OLS regression of LHS = 3 - (S_P1+S_P2+S_P3-CI) on x = S_P4 '
               'across 12 (defense, dataset) cells per dataset. '
               'C_1 = slope, C_2 = -intercept.'),
    'per_dataset': results,
    'pooled': {'C_1': float(slope), 'C_2': float(-intercept),
               'R_squared': float(r2), 'C_2_lower_envelope': float(c2_env),
               'n_cells': len(all_pts)},
    'per_dataset_C1_range': [float(min(c1_vals)), float(max(c1_vals))],
    'per_dataset_C2_range': [float(min(c2_vals)), float(max(c2_vals))],
}
with open('results/c1_c2_fits.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved -> results/c1_c2_fits.json')
