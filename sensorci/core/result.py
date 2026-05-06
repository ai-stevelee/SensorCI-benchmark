"""Result schemas: AttackResult, EvalResult."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class AttackResult:
    """Result from running one attack on (defense, dataset, seed) cell."""
    cell_id: str  # e.g. "tep_d5_a0_seed42"
    dataset: str
    defense: str
    attack: str
    seed: int
    metric_name: str
    metric_value: float
    metric_ci95: tuple[float, float] | None = None
    subscores: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    wall_seconds: float = 0.0
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("metric_ci95") is not None:
            d["metric_ci95"] = list(d["metric_ci95"])
        return d


@dataclass
class EvalResult:
    """Aggregated result across multiple AttackResults (e.g., 5 seeds)."""
    cell_id_prefix: str
    attack_results: list[AttackResult]
    summary: dict[str, float] = field(default_factory=dict)

    def add(self, r: AttackResult) -> None:
        self.attack_results.append(r)

    def aggregate(self) -> None:
        """Compute mean / std across attack_results."""
        if not self.attack_results:
            return
        vals = [r.metric_value for r in self.attack_results]
        self.summary = {
            "mean": float(sum(vals) / len(vals)),
            "std": float((sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5),
            "n_seeds": len(vals),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }


def save_results(results: list[AttackResult], out_path: Path) -> None:
    """Save results as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def load_results(path: Path) -> list[dict]:
    """Load results from JSONL (returns list of dicts)."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
