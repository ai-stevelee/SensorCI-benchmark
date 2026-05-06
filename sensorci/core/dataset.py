"""TSDataset: container for a SensorCI dataset.

Wraps personas + sharing matrix + role capability spec.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import json
from pathlib import Path
import numpy as np

from sensorci.core.persona import Persona, Segment, Role, ROLES, SharingLabel


@dataclass
class RoleCapability:
    """Per-role access capability (what the role can observe / what aux info they hold)."""
    role: Role
    can_observe_full_signal: bool
    downsample_factor: int  # 1 = no downsampling
    aux_description: str    # human-readable
    aux_data: dict | None = None  # role-specific prior knowledge


# Default role capabilities (from 02_dataset_curation.md §5.5)
DEFAULT_CAPABILITIES: dict[Role, RoleCapability] = {
    "Operator":  RoleCapability("Operator",  True,  1,  "Full dataset access"),
    "OEM":       RoleCapability("OEM",       False, 2,  "Aggregated fault stats per product line"),
    "Vendor":    RoleCapability("Vendor",    False, 4,  "Aggregate fault count + downsampled signals"),
    "Insurer":   RoleCapability("Insurer",   False, 8,  "Fleet-level annual statistics only"),
    "Regulator": RoleCapability("Regulator", False, 4,  "Incident-flagged segments only"),
    "Public":    RoleCapability("Public",    False, 16, "Published industry statistics"),
}


@dataclass
class TSDataset:
    """SensorCI dataset wrapper.

    Holds:
      - personas: list of Persona objects (each with multiple segments)
      - sharing_matrix: Phi_D mapping (role, attribute) -> S/N/P
      - role_capabilities: how each role observes signals + their aux info
      - dataset_metadata: documentation for LLM-judge prompting (datasheet, doc text)
    """
    name: str
    personas: list[Persona]
    sharing_matrix: dict[tuple[Role, str], SharingLabel] = field(default_factory=dict)
    role_capabilities: dict[Role, RoleCapability] = field(default_factory=lambda: dict(DEFAULT_CAPABILITIES))
    dataset_metadata: dict = field(default_factory=dict)
    class_list: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.personas)

    def iter_segments(self) -> Iterator[tuple[Persona, Segment]]:
        for p in self.personas:
            for s in p.segments:
                yield p, s

    def all_segments(self) -> list[tuple[Persona, Segment]]:
        return list(self.iter_segments())

    def attribute_set(self) -> set[str]:
        attrs: set[str] = set()
        for p in self.personas:
            attrs.update(p.all_attributes.keys())
        return attrs

    def not_share_attributes(self, role: Role) -> list[str]:
        """Attributes for which Phi_D(role, attr) = N (privacy-violating to disclose)."""
        return [
            a for (r, a), label in self.sharing_matrix.items()
            if r == role and label == "N"
        ]

    def share_attributes(self, role: Role) -> list[str]:
        return [
            a for (r, a), label in self.sharing_matrix.items()
            if r == role and label == "S"
        ]

    def save_metadata(self, path: Path) -> None:
        """Save schema metadata as JSON for inspection / reproduction."""
        out = {
            "name": self.name,
            "n_personas": len(self.personas),
            "n_segments_total": sum(len(p.segments) for p in self.personas),
            "class_list": self.class_list,
            "attribute_set": sorted(self.attribute_set()),
            "n_sharing_labels": len(self.sharing_matrix),
            "roles": list(ROLES),
            "dataset_metadata": self.dataset_metadata,
        }
        path.write_text(json.dumps(out, indent=2, default=str))


def default_sharing_matrix(attribute_categories: dict[str, str]) -> dict[tuple[Role, str], SharingLabel]:
    """Generate a default sharing matrix using simple heuristics.

    Heuristic (to be replaced by LLM-as-judge curation in Phase 3):
      - identity attributes: Operator=S, others=N (strict)
      - operating attributes: Operator=S, OEM=S, Vendor=P, Insurer=N, Regulator=N, Public=N
      - sensitive attributes: Operator=S, others=N

    This is a stub: real curation goes through `sensorci/judge/` (Phase 3).
    """
    matrix: dict[tuple[Role, str], SharingLabel] = {}
    for attr, cat in attribute_categories.items():
        for role in ROLES:
            if role == "Operator":
                matrix[(role, attr)] = "S"
            elif cat == "identity":
                matrix[(role, attr)] = "N"
            elif cat == "operating":
                if role == "OEM":
                    matrix[(role, attr)] = "S"
                elif role == "Vendor":
                    matrix[(role, attr)] = "P"
                else:
                    matrix[(role, attr)] = "N"
            else:  # sensitive
                matrix[(role, attr)] = "N"
    return matrix
