"""Matplotlib plotting utilities for SensorCI paper figures."""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise RuntimeError(
            "matplotlib not installed. Run: pip install matplotlib"
        )


def plot_pareto_frontier(
    defense_scores: dict[str, dict[str, float]],
    out_path: Path | str,
    x_axis: str = "S_P1",
    y_axis: str = "min_S_P3CI_S_P4",
    title: str = "Pareto Frontier of CI Quadrilemma",
) -> Path:
    """Scatter defenses in (P1, min(P3-CI, P4)) plane and draw frontier curve.

    defense_scores: dict defense -> {S_P1, S_P2, S_P3-CI, S_P4}
    """
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys, names = [], [], []
    for d, sc in defense_scores.items():
        xs.append(sc.get("S_P1", 0))
        ys.append(min(sc.get("S_P3-CI", 0), sc.get("S_P4", 0)))
        names.append(d)
    ax.scatter(xs, ys, s=120, alpha=0.75, c=range(len(xs)), cmap="tab10")
    for i, name in enumerate(names):
        ax.annotate(name, (xs[i], ys[i]), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    # Empirical Pareto frontier (upper-right boundary of dominant points)
    points = sorted(zip(xs, ys), key=lambda p: -p[0])
    frontier = []
    cur_y = -1
    for x, y in points:
        if y > cur_y:
            frontier.append((x, y))
            cur_y = y
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k--", lw=1.5, alpha=0.6, label="Empirical Pareto frontier")
    # Impossibility region shading
    ax.axhspan(0.8, 1.0, xmin=0.8, alpha=0.1, color="red", label="Target region (empty)")
    ax.set_xlabel(f"{x_axis} (utility)", fontsize=12)
    ax.set_ylabel(f"{y_axis} (privacy)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Pareto frontier to {out_path}")
    return out_path


def plot_radar_per_defense(
    defense_scores: dict[str, dict[str, float]],
    out_path: Path | str,
    title: str = "Per-Defense 4-Axis Radar",
) -> Path:
    """4-axis radar chart per defense (P1, P2, P3-CI, P4)."""
    plt = _ensure_matplotlib()
    axes_names = ["S_P1", "S_P2", "S_P3-CI", "S_P4"]
    n_defenses = len(defense_scores)
    cols = min(3, n_defenses)
    rows = (n_defenses + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             subplot_kw={"projection": "polar"})
    if n_defenses == 1:
        axs = [axs]
    elif rows == 1:
        axs = list(axs) if hasattr(axs, "__iter__") else [axs]
    else:
        axs = axs.flatten()

    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]

    for i, (name, sc) in enumerate(defense_scores.items()):
        if i >= len(axs):
            break
        ax = axs[i]
        vals = [sc.get(a, 0) for a in axes_names]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.25)
        ax.plot(angles, vals, "o-", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.5, 0.8])
        ax.set_title(name, fontsize=11, pad=10)
    # Hide empty subplots
    for j in range(n_defenses, len(axs)):
        axs[j].set_visible(False)
    fig.suptitle(title, fontsize=13)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_role_heatmap(
    per_role_scores: dict[str, dict[str, float]],
    out_path: Path | str,
    title: str = "Per-Role S_P3-CI Heatmap",
) -> Path:
    """Heatmap of defenses (rows) × roles (cols)."""
    plt = _ensure_matplotlib()
    defenses = list(per_role_scores.keys())
    if not defenses:
        return Path(out_path)
    roles = list(per_role_scores[defenses[0]].keys())
    M = np.array([[per_role_scores[d].get(r, 0.0) for r in roles] for d in defenses])
    fig, ax = plt.subplots(figsize=(max(6, len(roles)), max(4, len(defenses) * 0.4)))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=45, ha="right")
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels(defenses)
    for i in range(len(defenses)):
        for j in range(len(roles)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="S_P3-CI (higher = better)")
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_t2_curves(
    t2_per_defense: dict[str, dict[int, float]],
    out_path: Path | str,
    title: str = "T2: Cross-Segment Linkage AUC vs N",
) -> Path:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    for defense, curve in t2_per_defense.items():
        ns = sorted(curve.keys())
        aucs = [curve[n] for n in ns]
        ax.plot(ns, aucs, "o-", label=defense, lw=2, markersize=6)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("N segments per source")
    ax.set_ylabel("Linkage AUC")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_t4_curves(
    t4_per_defense: dict[str, dict[int, float]],
    out_path: Path | str,
    title: str = "T4: Longitudinal Cumulative Leakage",
) -> Path:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    for defense, curve in t4_per_defense.items():
        ts = sorted(curve.keys())
        leaks = [curve[t] for t in ts]
        ax.plot(ts, leaks, "o-", label=defense, lw=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative releases (days, log scale)")
    ax.set_ylabel("Cumulative leakage (1 - S_P3-CI)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_all_figures(
    defense_scores: dict[str, dict[str, float]],
    per_role_scores: dict[str, dict[str, float]] | None,
    t2_curves: dict[str, dict[int, float]] | None,
    t4_curves: dict[str, dict[int, float]] | None,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Generate all paper figures into out_dir/. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["pareto"] = plot_pareto_frontier(defense_scores, out_dir / "fig_pareto_frontier.png")
    paths["radar"] = plot_radar_per_defense(defense_scores, out_dir / "fig_radar_per_defense.png")
    if per_role_scores:
        paths["per_role"] = plot_per_role_heatmap(per_role_scores, out_dir / "fig_per_role_heatmap.png")
    if t2_curves:
        paths["t2"] = plot_t2_curves(t2_curves, out_dir / "fig_t2_curves.png")
    if t4_curves:
        paths["t4"] = plot_t4_curves(t4_curves, out_dir / "fig_t4_curves.png")
    return paths
