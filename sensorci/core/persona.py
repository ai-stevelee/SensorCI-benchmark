"""Persona, Segment, and Role definitions.

Implements the persona x segment x attribute schema described in
benchmark_session/02_dataset_curation.md and the 6-role taxonomy from
benchmark_session/01_problem_formulation.md (Definition 3).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np

# 6 stakeholder roles from EU Data Act Annex II
Role = Literal["Operator", "OEM", "Vendor", "Insurer", "Regulator", "Public"]
ROLES: tuple[Role, ...] = ("Operator", "OEM", "Vendor", "Insurer", "Regulator", "Public")

# Sharing label for Phi_D(role, attribute) -> {S, N, P}
SharingLabel = Literal["S", "N", "P"]


@dataclass
class Segment:
    """A single time-series segment from a persona.

    signal: shape (n_channels, segment_length)
    """
    segment_id: str
    signal: np.ndarray
    channels: list[str]
    sampling_rate_hz: float
    class_label: str | None = None
    release_time: int = 0  # for T4 longitudinal evaluation
    timestamp_index: int = 0  # within persona ordering

    def __post_init__(self):
        if self.signal.ndim != 2:
            raise ValueError(
                f"signal must be 2D (channels, length), got shape {self.signal.shape}"
            )
        if self.signal.shape[0] != len(self.channels):
            raise ValueError(
                f"signal channels {self.signal.shape[0]} != channels list {len(self.channels)}"
            )


@dataclass
class Persona:
    """A persona = source identity (bearing serial / engine unit / patient ID / ...).

    Holds attribute dicts at three sensitivity tiers:
      identity: directly identifies the persona (manufacturer, serial, location)
      operating: operating conditions (RPM, load, season)
      sensitive: confidential metadata (owner, deployment date, diagnosis)

    The CI sharing matrix Phi_D(role, attribute) determines which role may
    learn each attribute.
    """
    persona_id: str
    dataset: str
    identity_attributes: dict[str, Any] = field(default_factory=dict)
    operating_attributes: dict[str, Any] = field(default_factory=dict)
    sensitive_attributes: dict[str, Any] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)

    @property
    def all_attributes(self) -> dict[str, Any]:
        """Flattened attribute dict (identity + operating + sensitive)."""
        return {
            **self.identity_attributes,
            **self.operating_attributes,
            **self.sensitive_attributes,
        }

    @property
    def attribute_categories(self) -> dict[str, str]:
        """Map attribute name -> category ('identity' / 'operating' / 'sensitive')."""
        cats: dict[str, str] = {}
        for k in self.identity_attributes:
            cats[k] = "identity"
        for k in self.operating_attributes:
            cats[k] = "operating"
        for k in self.sensitive_attributes:
            cats[k] = "sensitive"
        return cats

    def sample_segment(self, rng: np.random.Generator | None = None) -> Segment:
        rng = rng or np.random.default_rng()
        return self.segments[int(rng.integers(0, len(self.segments)))]
