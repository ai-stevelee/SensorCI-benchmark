"""Contamination control via counterfactual relabeling and metadata blinding.

Addresses Reviewer Q4: How do we ensure A0 results aren't dominated by
LLM prior memorization of well-known datasets?

Strategy:
  1. Counterfactual relabeling: rename dataset, channels, fault types to
     semantically-equivalent but textually-novel strings.
  2. Metadata blinding: strip dataset name + citation text from prompts.
  3. Contamination delta: run A0 on (a) original prompts (b) blinded prompts;
     measure accuracy drop. Drop = leakage from prior memorization.

This isolates "leakage from signal information" vs "leakage from training-set
memorization."
"""

from __future__ import annotations
import re
import hashlib
from copy import deepcopy
from dataclasses import dataclass

from sensorci.core.dataset import TSDataset
from sensorci.core.persona import Persona, Segment


def _stable_alias(original: str, prefix: str = "X") -> str:
    """Generate a deterministic alias from a string (so reruns produce same alias)."""
    h = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"


@dataclass
class RelabelMap:
    """Bidirectional mapping for counterfactual relabeling."""
    dataset_name: str  # original -> alias
    channel_aliases: dict[str, str]
    class_aliases: dict[str, str]
    attribute_value_aliases: dict[str, dict] = None  # attr -> {value: alias}

    def to_alias(self, kind: str, value: str) -> str:
        if kind == "dataset":
            return self.dataset_name
        if kind == "channel":
            return self.channel_aliases.get(value, value)
        if kind == "class":
            return self.class_aliases.get(value, value)
        return value


def counterfactual_relabel(
    dataset: TSDataset,
    seed_str: str = "sensorci_cf_v1",
) -> tuple[TSDataset, RelabelMap]:
    """Return a new TSDataset with relabeled (dataset name, channels, classes,
    selected attribute values) but identical signal data.

    The signal arrays are NOT modified; only metadata text is replaced.
    Use this for A0 contamination probing: A0 accuracy on the original vs
    counterfactual dataset measures how much the LLM relies on names vs signals.
    """
    # Generate aliases
    dataset_alias = _stable_alias(f"{seed_str}_{dataset.name}", prefix="DSET")
    channel_aliases = {
        ch: _stable_alias(f"{seed_str}_{dataset.name}_{ch}", prefix="ch")
        for ch in (dataset.dataset_metadata.get("channels") or [])
    }
    class_aliases = {
        cl: _stable_alias(f"{seed_str}_{dataset.name}_{cl}", prefix="cls")
        for cl in dataset.class_list
    }

    # Selectively alias attribute values for identity attributes
    attribute_value_aliases: dict[str, dict] = {}
    if dataset.personas:
        for attr in dataset.personas[0].identity_attributes:
            seen = set()
            attribute_value_aliases[attr] = {}
            for p in dataset.personas:
                v = p.identity_attributes.get(attr)
                if v is None or v in seen:
                    continue
                seen.add(v)
                attribute_value_aliases[attr][v] = _stable_alias(
                    f"{seed_str}_{attr}_{v}", prefix=attr[:3].upper()
                )

    relabel_map = RelabelMap(
        dataset_name=dataset_alias,
        channel_aliases=channel_aliases,
        class_aliases=class_aliases,
        attribute_value_aliases=attribute_value_aliases,
    )

    # Build new dataset
    new_personas: list[Persona] = []
    for p in dataset.personas:
        new_segs = []
        for s in p.segments:
            new_segs.append(Segment(
                segment_id=s.segment_id,
                signal=s.signal,  # SIGNAL UNCHANGED
                channels=[channel_aliases.get(c, c) for c in s.channels],
                sampling_rate_hz=s.sampling_rate_hz,
                class_label=class_aliases.get(s.class_label, s.class_label) if s.class_label else None,
                release_time=s.release_time,
                timestamp_index=s.timestamp_index,
            ))
        new_identity = {
            k: attribute_value_aliases.get(k, {}).get(v, v)
            for k, v in p.identity_attributes.items()
        }
        new_personas.append(Persona(
            persona_id=p.persona_id,  # IDs preserved (intra-dataset matching)
            dataset=dataset_alias,
            identity_attributes=new_identity,
            operating_attributes=deepcopy(p.operating_attributes),
            sensitive_attributes=deepcopy(p.sensitive_attributes),
            segments=new_segs,
        ))

    new_meta = deepcopy(dataset.dataset_metadata)
    new_meta["domain"] = "Industrial sensor data (anonymized for contamination probe)"
    new_meta["citation"] = "(blinded)"
    if "channels" in new_meta:
        new_meta["channels"] = [channel_aliases.get(c, c) for c in new_meta["channels"]]

    new_dataset = TSDataset(
        name=dataset_alias,
        personas=new_personas,
        sharing_matrix={
            (r, a): label for (r, a), label in dataset.sharing_matrix.items()
        },
        role_capabilities=dataset.role_capabilities,
        dataset_metadata=new_meta,
        class_list=[class_aliases.get(c, c) for c in dataset.class_list],
    )
    return new_dataset, relabel_map


def metadata_blind(prompt_text: str, dataset_name: str,
                   blind_keywords: list[str] | None = None) -> str:
    """Strip dataset-identifying keywords from a prompt text.

    Lighter-weight than counterfactual_relabel: only modifies prompt strings,
    not dataset object. Useful for ad-hoc per-prompt blinding.
    """
    blind_keywords = blind_keywords or [dataset_name]
    out = prompt_text
    for kw in blind_keywords:
        # Case-insensitive replace with placeholder
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        out = pattern.sub("[BLINDED_DATASET]", out)
    # Remove typical citation patterns
    out = re.sub(r"\(([A-Z][a-z]+(?:\s+(?:&|et\s+al\.?))?\s+\d{4})\)", "(citation)", out)
    return out


def contamination_delta(
    original_accuracy: dict[str, float],
    blinded_accuracy: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute per-attribute delta: original_acc - blinded_acc.

    Positive delta = LLM accuracy drops when metadata is blinded
                     (suggests the LLM relied on prior knowledge).
    Near-zero delta = LLM relies on signal content (good for SensorCI validity).

    Args:
      original_accuracy: dict attr_name -> A0 accuracy with original prompts
      blinded_accuracy:  dict attr_name -> A0 accuracy with blinded prompts

    Returns:
      Per-attribute analysis with delta + interpretation flag.
    """
    out = {}
    for attr in original_accuracy:
        if attr not in blinded_accuracy:
            continue
        orig = original_accuracy[attr]
        blind = blinded_accuracy[attr]
        delta = orig - blind
        out[attr] = {
            "original_acc": orig,
            "blinded_acc": blind,
            "delta": delta,
            "contamination_flag": "high" if delta > 0.15 else ("moderate" if delta > 0.05 else "low"),
        }
    return out
