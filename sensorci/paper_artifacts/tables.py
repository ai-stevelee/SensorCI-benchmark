"""Generate LaTeX tables for the SensorCI paper.

Each function returns a string containing the booktabs-style LaTeX table.
write_latex_table() saves to file.
"""

from __future__ import annotations
from pathlib import Path


def write_latex_table(table_str: str, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_str, encoding="utf-8")
    return out_path


def make_headline_table(
    a0_results: dict[str, dict[str, float]],
    target_threshold: float = 0.05,
    caption: str = "$A_0$ frontier LLM source identification accuracy.",
    label: str = "tab:headline",
) -> str:
    """Headline A0 table: defense × LLM × accuracy.

    a0_results: dict defense -> {llm_name -> accuracy_in_0_1}
    """
    if not a0_results:
        return ""
    defenses = list(a0_results.keys())
    llms = list(a0_results[defenses[0]].keys())

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(llms) + 1) + "c}",
        r"\toprule",
        "Defense & " + " & ".join(llms) + r" & \textbf{Mean} & Target met? \\",
        r"\midrule",
    ]
    for d in defenses:
        row = [d]
        vals = [a0_results[d].get(l, 0.0) for l in llms]
        for v in vals:
            row.append(f"{100 * v:.1f}")
        mean_val = sum(vals) / len(vals) if vals else 0.0
        row.append(f"\\textbf{{{100 * mean_val:.1f}}}")
        row.append(r"$\checkmark$" if mean_val <= target_threshold else r"$\times$")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_per_role_table(
    per_role_scores: dict[str, dict[str, float]],
    caption: str = "Per-role $S_{P_3\\text{-CI}}$ by defense (higher = better).",
    label: str = "tab:per_role",
) -> str:
    if not per_role_scores:
        return ""
    defenses = list(per_role_scores.keys())
    roles = list(per_role_scores[defenses[0]].keys())
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * (len(roles) + 1) + "}",
        r"\toprule",
        "Defense & " + " & ".join(roles) + r" & \textbf{Worst} \\",
        r"\midrule",
    ]
    for d in defenses:
        scores = per_role_scores[d]
        row = [d]
        vals = [scores.get(r, 0.0) for r in roles]
        for v in vals:
            row.append(f"{v:.2f}")
        worst = min(vals) if vals else 0.0
        row.append(f"\\textbf{{{worst:.2f}}}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_composite_table(
    composite_per_defense: dict[str, dict[str, float]],
    caption: str = "Cross-dataset composite scores.",
    label: str = "tab:composite",
) -> str:
    """composite_per_defense: defense -> {S_P1, S_P2, S_P3-CI, S_P4, HM, WA, WR, rank}"""
    if not composite_per_defense:
        return ""
    defenses = list(composite_per_defense.keys())
    cols = ["S_P1", "S_P2", "S_P3-CI", "S_P4", "HM", "WA", "WR", "Rank"]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Defense & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\",
        r"\midrule",
    ]
    for d in defenses:
        sc = composite_per_defense[d]
        row = [d]
        for c in cols:
            v = sc.get(c, 0.0)
            row.append(f"{v:.2f}" if c != "Rank" else f"{v:.1f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
