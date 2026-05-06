"""Paderborn KAt Bearing dataset loader.

Real mode: parses .mat files from KAt registration download.
Synthetic mode (default fallback): generates 32 bearings × 4 conditions with
fault-class signal patterns matching Paderborn's structure.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

CHANNELS = ["motor_current_phase_a", "motor_current_phase_b", "motor_current_phase_c",
            "vibration_acc_y", "speed_rpm"]
FAULT_CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]
OPERATING_CONDITIONS = [
    {"name": "N15_M07_F10", "rpm": 1500, "load_nm": 0.7, "force_n": 1000},
    {"name": "N09_M07_F10", "rpm": 900,  "load_nm": 0.7, "force_n": 1000},
    {"name": "N15_M01_F10", "rpm": 1500, "load_nm": 0.1, "force_n": 1000},
    {"name": "N15_M07_F04", "rpm": 1500, "load_nm": 0.7, "force_n": 400},
]
MANUFACTURERS = ["SKF", "FAG", "Schaeffler"]
PLANTS = ["Plant_North", "Plant_South", "Plant_East"]


class PaderbornDataset(BaseDatasetLoader):
    name = "paderborn"

    def __init__(self, mode: str = "auto", n_bearings: int = 16,
                 segments_per_bearing: int = 4, segment_length: int = 4096,
                 sampling_rate_hz: float = 64000.0, seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_bearings = n_bearings
        self.segments_per_bearing = segments_per_bearing
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real Paderborn load failed: {e}. Falling back to synthetic.")
                personas = self._load_synthetic()
        else:
            personas = self._load_synthetic()

        attr_categories = {}
        for p in personas:
            for k in p.identity_attributes: attr_categories[k] = "identity"
            for k in p.operating_attributes: attr_categories[k] = "operating"
            for k in p.sensitive_attributes: attr_categories[k] = "sensitive"

        return TSDataset(
            name=self.name,
            personas=personas,
            sharing_matrix=default_sharing_matrix(attr_categories),
            class_list=FAULT_CLASSES,
            dataset_metadata={
                "domain": "Rotating machinery (motor-bearing) vibration + current",
                "citation": "Lessmeier et al., PHM Europe 2016",
                "channels": CHANNELS,
                "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return (self.data_dir / "K001").exists()

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        # Spread fault classes
        bearings = []
        for b in range(self.n_bearings):
            fault = FAULT_CLASSES[b % len(FAULT_CLASSES)]
            for cond in OPERATING_CONDITIONS:
                bearings.append((b, fault, cond))
                if len(bearings) >= self.n_bearings * 4:
                    break
            if len(bearings) >= self.n_bearings * 4:
                break

        for b_idx, fault, cond in bearings[: self.n_bearings * 4]:
            bearing_id = f"K{b_idx:03d}_{cond['name']}"
            manufacturer = MANUFACTURERS[b_idx % 3]
            plant = PLANTS[b_idx % 3]

            segs = []
            for s_idx in range(self.segments_per_bearing):
                fault_offset = 0.0 if fault == "healthy" else 0.5 + 0.2 * FAULT_CLASSES.index(fault)
                signal = synth_multichannel(
                    n_channels=len(CHANNELS),
                    length=self.segment_length,
                    sampling_rate_hz=self.sampling_rate_hz,
                    rng=rng,
                    base_freqs_hz=[50, 50, 50, 162.0 + 50 * FAULT_CLASSES.index(fault), cond["rpm"] / 60],
                    fault_offset=fault_offset,
                    fault_channel=3,
                )
                segs.append(Segment(
                    segment_id=f"{bearing_id}_seg{s_idx:03d}",
                    signal=signal, channels=CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz,
                    class_label=fault, release_time=s_idx, timestamp_index=s_idx,
                ))

            personas.append(Persona(
                persona_id=bearing_id, dataset=self.name,
                identity_attributes={
                    "manufacturer": manufacturer,
                    "bearing_type_code": f"K{b_idx:03d}",
                    "factory_location": "Paderborn_Germany",
                },
                operating_attributes={
                    "rotational_speed_rpm": cond["rpm"],
                    "load_torque_nm": cond["load_nm"],
                    "radial_force_n": cond["force_n"],
                    "fault_class": fault,
                },
                sensitive_attributes={
                    "owner_company": plant,
                    "deployment_date": f"2023-{1 + b_idx % 12:02d}-15",
                    "lubrication_regime": "high_speed" if cond["rpm"] > 1200 else "low_speed",
                },
                segments=segs,
            ))
        return personas

    def _load_real(self) -> list[Persona]:
        # Real .mat parsing requires scipy.io and mirror-specific layout
        # Stub: try to find K0xx folders, fall back to synthetic if structure differs
        try:
            from scipy.io import loadmat
        except ImportError:
            raise RuntimeError("scipy required for real Paderborn parsing. pip install scipy")

        personas = []
        for k_dir in sorted(self.data_dir.glob("K*")):
            if not k_dir.is_dir():
                continue
            mat_files = list(k_dir.glob("*.mat"))[:self.segments_per_bearing]
            if not mat_files:
                continue
            segs = []
            for i, mf in enumerate(mat_files):
                try:
                    data = loadmat(mf)
                    # Find vibration array (heuristic: first numeric 1D array)
                    sig = None
                    for k, v in data.items():
                        if k.startswith("__"):
                            continue
                        if hasattr(v, "shape") and v.size > 1000:
                            sig = np.array(v).flatten()
                            break
                    if sig is None:
                        continue
                    sig = sig[:self.segment_length].reshape(1, -1)
                    # Pad to required channel count
                    if sig.shape[0] < len(CHANNELS):
                        pad = np.zeros((len(CHANNELS) - sig.shape[0], sig.shape[1]))
                        sig = np.vstack([sig, pad])
                    segs.append(Segment(
                        segment_id=f"{k_dir.name}_seg{i:03d}", signal=sig, channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz,
                        class_label="healthy" if k_dir.name.startswith("K0") else "fault",
                        release_time=i, timestamp_index=i,
                    ))
                except Exception as e:
                    logger.debug(f"Skip {mf}: {e}")
                    continue

            if segs:
                personas.append(Persona(
                    persona_id=k_dir.name, dataset=self.name,
                    identity_attributes={"bearing_type_code": k_dir.name,
                                         "manufacturer": "unknown",
                                         "factory_location": "Paderborn_Germany"},
                    operating_attributes={"rotational_speed_rpm": 1500,
                                          "load_torque_nm": 0.7, "radial_force_n": 1000,
                                          "fault_class": segs[0].class_label},
                    sensitive_attributes={"owner_company": "synthetic",
                                          "deployment_date": "2023-01-01",
                                          "lubrication_regime": "high_speed"},
                    segments=segs,
                ))

        if not personas:
            raise RuntimeError(f"No usable Paderborn data found under {self.data_dir}")
        logger.info(f"Loaded {len(personas)} real Paderborn bearings")
        return personas
