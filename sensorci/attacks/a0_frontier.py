"""A0: Frontier LLM Inference Attack (★ headline attack of SensorCI).

For each transformed signal, encode as text and ask 1-3 frontier LLMs to identify:
  - source persona ID (1-of-N classification)
  - class label (e.g., fault type)
  - operating condition
  - sensitive attributes (manufacturer family, owner, etc.)

Adversary advantage:
    Adv = Acc(LLM) - Acc(prior chance)
"""

from __future__ import annotations
import json
import re
import time
import logging
from typing import Any
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.defenses.base import BaseDefense
from sensorci.encodings import ENCODERS
from sensorci.api_pool import get_default_pool, APIKeyPool

logger = logging.getLogger(__name__)


# Prompt templates — addresses Reviewer Q2 (A0 prompt sensitivity).
# Use --prompt_variant CLI flag to switch.

PROMPT_TEMPLATES = {
    # V0: zero-shot, current default
    "v0_zeroshot": """\
You are a senior industrial sensor analyst. Below is a sensor segment from
the {dataset_name} dataset ({domain_description}) after a privacy-preserving
transformation. Your task is to infer the source and operating context.

Encoding type: {encoding_type}
Channels: {channel_list}
Sampling rate: {sampling_rate_hz} Hz

Signal data:
{encoded_signal}

Choose your best guess for each of the following. Respond ONLY with a valid JSON object.
Each value must be one of the listed candidates if a list is given.

{question_block}

JSON response (no markdown, no explanation):
""",

    # V1: chain-of-thought (with reasoning preamble)
    "v1_cot": """\
You are a senior industrial sensor analyst. Below is a sensor segment from
the {dataset_name} dataset ({domain_description}) after a privacy-preserving
transformation.

Encoding type: {encoding_type}
Channels: {channel_list}
Sampling rate: {sampling_rate_hz} Hz

Signal data:
{encoded_signal}

First, reason briefly (3-5 sentences) about: (a) which spectral or statistical
features are most diagnostic, (b) what the transformation might have altered,
(c) which fields you can most/least confidently predict.

Then, on a new line starting with "JSON:", output a single valid JSON object
with these fields:

{question_block}
""",

    # V2: minimal prompt (no domain hint), tests whether priming matters
    "v2_minimal": """\
Sensor segment ({encoding_type} encoding):
{encoded_signal}

Identify the following. Respond as a single JSON object only.
{question_block}
""",

    # V3: blinded — addresses Reviewer Q4 (contamination control)
    "v3_blinded": """\
Below is a sensor segment from an industrial dataset (identity withheld) after
a privacy-preserving transformation.

Encoding: {encoding_type}
Channel labels: {channel_list}
Sampling rate: {sampling_rate_hz} Hz

Signal data:
{encoded_signal}

Choose the best guess for each field. Respond ONLY with a valid JSON object.
{question_block}

JSON response:
""",
}

# Backward-compat alias
PROMPT_TEMPLATE = PROMPT_TEMPLATES["v0_zeroshot"]


def _extract_json(text: str) -> dict | None:
    """Try to pull a JSON object out of LLM response text."""
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _build_question_block(
    candidate_personas: list[str],
    class_list: list[str],
    candidate_attributes: dict[str, list[str]],
) -> str:
    lines = []
    if candidate_personas:
        ps = ", ".join(candidate_personas[:30])  # limit prompt size
        suffix = "" if len(candidate_personas) <= 30 else f" (...and {len(candidate_personas)-30} more)"
        lines.append(f'  "source_persona_id" — choose one of: [{ps}{suffix}]')
    if class_list:
        lines.append(f'  "class_label" — choose one of: [{", ".join(class_list)}]')
    for attr, cands in candidate_attributes.items():
        if cands:
            lines.append(f'  "{attr}" — choose one of: [{", ".join(map(str, cands[:20]))}]')
        else:
            lines.append(f'  "{attr}" — your best free-form guess')
    lines.append('  "confidence" — float 0-1 indicating overall confidence')
    return "Fields:\n" + "\n".join(lines)


def _env_model_list() -> list[str]:
    """Build full model list from .env (AZURE_OPENAI_MODEL / CLAUDE_MODEL / GEMINI_MODEL)."""
    import os
    models = []
    for var in ("AZURE_OPENAI_MODEL", "CLAUDE_MODEL", "GEMINI_MODEL"):
        val = os.environ.get(var, "").strip()
        if val:
            models += [m.strip() for m in val.split(",") if m.strip()]
    return models


def _env_cheap_models() -> list[str]:
    """One cheap model per provider (first in each list)."""
    import os
    cheap = []
    for var in ("AZURE_OPENAI_MODEL", "CLAUDE_MODEL", "GEMINI_MODEL"):
        val = os.environ.get(var, "").strip()
        if val:
            first = val.split(",")[0].strip()
            if first:
                cheap.append(first)
    return cheap or ["gpt-5.4", "claude-sonnet-4-5", "gemini-2.5-flash"]


class FrontierLLMAttack(BaseAttack):
    name = "a0_frontier"
    target_property = "P3-CI"  # primary; also touches P4 indirectly

    # Fallback model groups (overridden at runtime by env vars when track is used)
    PROPRIETARY_MODELS: list[str] = []   # populated from env
    CHEAP_MODELS: list[str] = []         # populated from env
    OPEN_MODELS = [                      # not available on API hub; kept for API compatibility
        "meta-llama/llama-4-behemoth",
        "mistralai/Mistral-Large-2026",
        "Qwen/Qwen2.5-72B-Instruct",
    ]

    def __init__(
        self,
        models: list[str] | None = None,
        encodings: list[str] | None = None,
        prompt_variant: str = "v0_zeroshot",
        max_workers: int = 32,
        max_tokens: int = 40000,
        target_attributes: list[str] | None = None,
        track: str | None = None,  # "proprietary" / "open" / "cheap" / None (use models arg)
    ):
        super().__init__()
        # Resolve track model lists from env at instantiation time
        env_all = _env_model_list()
        env_cheap = _env_cheap_models()
        if track is not None:
            track_models = {
                "proprietary": env_all or self.PROPRIETARY_MODELS,
                "open": self.OPEN_MODELS,
                "cheap": env_cheap or self.CHEAP_MODELS,
            }
            if track not in track_models:
                raise ValueError(f"track must be one of {list(track_models)}, got {track!r}")
            self.models = track_models[track]
        else:
            self.models = models or env_all or ["gpt-5.4", "claude-sonnet-4-5", "gemini-2.5-flash"]
        self.encodings = encodings or ["raw_stats"]
        if prompt_variant not in PROMPT_TEMPLATES:
            raise ValueError(f"prompt_variant must be one of {list(PROMPT_TEMPLATES)}")
        self.prompt_variant = prompt_variant
        self.prompt_template = PROMPT_TEMPLATES[prompt_variant]
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        # Which sensitive attributes to probe; defaults to all sensitive_attributes from personas
        self.target_attributes = target_attributes
        self.track = track

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        pool: APIKeyPool | None = None,
        **kwargs,
    ) -> AttackResult:
        pool = pool or get_default_pool()
        rng = np.random.default_rng(seed)

        # Sample (persona, segment) pairs
        all_segs = dataset.all_segments()
        if len(all_segs) > n_samples:
            idxs = rng.choice(len(all_segs), size=n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs

        # Determine candidate sets
        candidate_personas = [p.persona_id for p in dataset.personas]
        class_list = dataset.class_list

        # Determine which attributes to probe
        if self.target_attributes is None:
            # Default: all sensitive + identity attributes from first persona
            sample_persona = dataset.personas[0]
            target_attrs = list(sample_persona.identity_attributes.keys()) + \
                           list(sample_persona.sensitive_attributes.keys())
        else:
            target_attrs = list(self.target_attributes)

        # Pre-compute candidate values for each attribute (union across personas)
        candidate_attributes: dict[str, list[str]] = {}
        for attr in target_attrs:
            vals = set()
            for p in dataset.personas:
                v = p.all_attributes.get(attr)
                if v is not None:
                    vals.add(str(v))
            candidate_attributes[attr] = sorted(vals)

        question_block = _build_question_block(candidate_personas, class_list, candidate_attributes)

        # Build prompts: one per (sample, encoding)
        prompts = []  # list of (model, messages, ground_truth_dict)
        for persona, seg in sampled:
            transformed = defense.transform(
                seg.signal, source_id=persona.persona_id, segment_id=seg.segment_id
            )
            for enc_name in self.encodings:
                encoder = ENCODERS[enc_name]
                encoded = encoder(
                    transformed,
                    sampling_rate_hz=seg.sampling_rate_hz,
                    channels=seg.channels,
                )
                prompt_text = self.prompt_template.format(
                    dataset_name=dataset.name,
                    domain_description=dataset.dataset_metadata.get("domain", ""),
                    encoding_type=enc_name,
                    channel_list=", ".join(seg.channels),
                    sampling_rate_hz=seg.sampling_rate_hz,
                    encoded_signal=encoded,
                    question_block=question_block,
                )
                ground_truth = {
                    "source_persona_id": persona.persona_id,
                    "class_label": seg.class_label,
                    **{k: str(v) for k, v in persona.all_attributes.items() if k in target_attrs},
                }
                prompts.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "ground_truth": ground_truth,
                    "encoding": enc_name,
                    "persona_id": persona.persona_id,
                    "segment_id": seg.segment_id,
                })

        # Run per model
        per_model_results: dict[str, list[dict]] = {}
        t0 = time.time()
        for model in self.models:
            logger.info(f"Running A0 with {model} on {len(prompts)} prompts...")
            messages_list = [p["messages"] for p in prompts]
            responses = pool.batch_call(
                model, messages_list,
                max_workers=self.max_workers,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            per_model_results[model] = []
            for prompt, resp in zip(prompts, responses):
                parsed = _extract_json(resp.text) if resp.text else None
                per_model_results[model].append({
                    "ground_truth": prompt["ground_truth"],
                    "predicted": parsed,
                    "encoding": prompt["encoding"],
                    "raw_text": resp.text[:500],
                    "error": resp.error,
                    "elapsed_s": resp.elapsed_s,
                })
        elapsed = time.time() - t0

        # Compute per-attribute accuracy and Adv
        attrs_to_score = ["source_persona_id", "class_label"] + target_attrs
        per_model_scores: dict[str, dict[str, float]] = {}
        for model, results in per_model_results.items():
            scores: dict[str, dict[str, float]] = {}
            for attr in attrs_to_score:
                correct, total, parsed_n = 0, 0, 0
                for r in results:
                    gt_val = str(r["ground_truth"].get(attr, "")).strip().lower()
                    if not gt_val:
                        continue
                    total += 1
                    if r["predicted"] is None:
                        continue
                    parsed_n += 1
                    pred_val = str(r["predicted"].get(attr, "")).strip().lower()
                    if pred_val == gt_val:
                        correct += 1
                acc = correct / max(total, 1)
                # Prior accuracy = 1/|candidates| for categorical
                if attr == "source_persona_id":
                    prior = 1.0 / max(len(candidate_personas), 1)
                elif attr == "class_label":
                    prior = 1.0 / max(len(class_list), 1)
                else:
                    prior = 1.0 / max(len(candidate_attributes.get(attr, ["_"])), 1)
                adv = max(acc - prior, 0.0)
                scores[attr] = {"acc": acc, "prior": prior, "adv": adv,
                                "parsed_rate": parsed_n / max(total, 1)}
            per_model_scores[model] = scores

        # Aggregate: max Adv across (model, attribute) is the headline
        max_adv = 0.0
        max_cell = ""
        for model, scores in per_model_scores.items():
            for attr, s in scores.items():
                if s["adv"] > max_adv:
                    max_adv = s["adv"]
                    max_cell = f"{model}/{attr}"

        cell_id = f"{dataset.name}_{defense.name}_{self.name}_seed{seed}"
        return AttackResult(
            cell_id=cell_id,
            dataset=dataset.name,
            defense=defense.name,
            attack=self.name,
            seed=seed,
            metric_name="adv_a0",
            metric_value=max_adv,
            subscores={"max_adv": max_adv},
            n_samples=len(sampled),
            wall_seconds=elapsed,
            extra={
                "per_model_scores": per_model_scores,
                "max_cell_label": max_cell,
                "n_prompts_total": len(prompts),
                "models": self.models,
                "encodings": self.encodings,
                "prompt_variant": self.prompt_variant,
                "track": self.track,
            },
        )
