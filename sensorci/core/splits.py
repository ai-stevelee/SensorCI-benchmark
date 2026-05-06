"""Persona-stratified train/val/test splits.

Critical for paper-quality evaluation:
  - Same persona MUST NOT appear in multiple splits (prevents source leak).
  - Random segment-level splits are FORBIDDEN since segments of one persona
    share defense keys, making cross-split correlation trivial.

Usage:
  splitter = PersonaStratifiedSplitter(train=0.6, val=0.2, test=0.2, seed=42)
  splits = splitter.split(dataset)
  # splits.train_segments / val_segments / test_segments are persona-disjoint
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sensorci.core.dataset import TSDataset
from sensorci.core.persona import Persona, Segment


@dataclass
class DatasetSplit:
    """Persona-disjoint split with segment lists for each phase."""
    train_personas: list[Persona]
    val_personas: list[Persona]
    test_personas: list[Persona]
    train_segments: list[tuple[Persona, Segment]]
    val_segments: list[tuple[Persona, Segment]]
    test_segments: list[tuple[Persona, Segment]]
    seed: int

    def __repr__(self) -> str:
        return (f"DatasetSplit(train={len(self.train_personas)}p/"
                f"{len(self.train_segments)}s, "
                f"val={len(self.val_personas)}p/{len(self.val_segments)}s, "
                f"test={len(self.test_personas)}p/{len(self.test_segments)}s)")


class PersonaStratifiedSplitter:
    """Persona-disjoint splitter.

    Splits personas into train/val/test groups such that no persona's segments
    appear in more than one phase. Maintains class-label balance where possible.
    """

    def __init__(
        self,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        seed: int = 42,
        balance_classes: bool = True,
    ):
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("train + val + test fractions must sum to 1.0")
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.balance_classes = balance_classes

    def split(self, dataset: TSDataset) -> DatasetSplit:
        rng = np.random.default_rng(self.seed)
        personas = list(dataset.personas)
        n = len(personas)
        if n < 3:
            raise ValueError(f"Need >=3 personas for train/val/test split, got {n}")

        # Optionally stratify by primary class label of first segment
        if self.balance_classes:
            by_class: dict[str, list[Persona]] = {}
            for p in personas:
                if not p.segments:
                    continue
                cls = p.segments[0].class_label or "unknown"
                by_class.setdefault(cls, []).append(p)
            train_p, val_p, test_p = [], [], []
            for cls, plist in by_class.items():
                rng.shuffle(plist)
                n_cls = len(plist)
                n_tr = max(1, int(n_cls * self.train_frac))
                n_va = max(0, int(n_cls * self.val_frac))
                train_p.extend(plist[:n_tr])
                val_p.extend(plist[n_tr:n_tr + n_va])
                test_p.extend(plist[n_tr + n_va:])
        else:
            shuffled = list(personas)
            rng.shuffle(shuffled)
            n_tr = max(1, int(n * self.train_frac))
            n_va = max(0, int(n * self.val_frac))
            train_p = shuffled[:n_tr]
            val_p = shuffled[n_tr:n_tr + n_va]
            test_p = shuffled[n_tr + n_va:]

        # Ensure no empty splits if possible
        if not val_p and len(train_p) > 1:
            val_p = [train_p.pop()]
        if not test_p and len(train_p) > 1:
            test_p = [train_p.pop()]

        train_segs = [(p, s) for p in train_p for s in p.segments]
        val_segs = [(p, s) for p in val_p for s in p.segments]
        test_segs = [(p, s) for p in test_p for s in p.segments]

        return DatasetSplit(
            train_personas=train_p, val_personas=val_p, test_personas=test_p,
            train_segments=train_segs, val_segments=val_segs, test_segments=test_segs,
            seed=self.seed,
        )

    @staticmethod
    def assert_no_leak(split: DatasetSplit) -> None:
        """Verify no persona appears in multiple phases."""
        train_ids = {p.persona_id for p in split.train_personas}
        val_ids = {p.persona_id for p in split.val_personas}
        test_ids = {p.persona_id for p in split.test_personas}
        if train_ids & val_ids:
            raise ValueError(f"Persona leak train↔val: {train_ids & val_ids}")
        if train_ids & test_ids:
            raise ValueError(f"Persona leak train↔test: {train_ids & test_ids}")
        if val_ids & test_ids:
            raise ValueError(f"Persona leak val↔test: {val_ids & test_ids}")
