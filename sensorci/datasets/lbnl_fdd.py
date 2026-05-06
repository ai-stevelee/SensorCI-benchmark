"""LBNL ASHRAE FDD loader. Real (CSV) + synthetic fallback."""

from __future__ import annotations
import logging
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

CHANNELS = ["oat", "mat", "sat", "rat", "fan_supply_pwr", "fan_return_pwr", "damper_pos", "valve_pos"]
FAULT_CLASSES = ["normal", "damper_stuck", "valve_stuck", "sensor_bias",
                 "fan_failure", "leak", "controller_drift", "schedule_violation"]
BUILDING_TYPES = ["office", "school", "hospital", "retail"]
OWNERS = ["University", "Corp_X", "Municipal", "Healthcare_Org"]
LOCATIONS = ["Berkeley_CA", "Chicago_IL", "Boston_MA", "Atlanta_GA"]


class LBNLFDDDataset(BaseDatasetLoader):
    name = "lbnl_fdd"

    def __init__(self, mode: str = "auto", n_buildings: int = 8, ahus_per_building: int = 2,
                 segments_per_ahu: int = 6, segment_length: int = 1440,  # 1 day at 1-min
                 sampling_rate_hz: float = 1 / 60, seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_buildings = n_buildings
        self.ahus_per_building = ahus_per_building
        self.segments_per_ahu = segments_per_ahu
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real LBNL load failed: {e}. Falling back to synthetic.")
                personas = self._load_synthetic()
        else:
            personas = self._load_synthetic()

        attr_categories = {}
        for p in personas:
            for k in p.identity_attributes: attr_categories[k] = "identity"
            for k in p.operating_attributes: attr_categories[k] = "operating"
            for k in p.sensitive_attributes: attr_categories[k] = "sensitive"

        return TSDataset(
            name=self.name, personas=personas,
            sharing_matrix=default_sharing_matrix(attr_categories),
            class_list=FAULT_CLASSES,
            dataset_metadata={
                "domain": "HVAC building fault detection",
                "citation": "Granderson et al., Scientific Data 2020",
                "channels": CHANNELS, "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return (self.data_dir / "AHU").exists()

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        for b_idx in range(self.n_buildings):
            for a_idx in range(self.ahus_per_building):
                bid = f"BLDG-{b_idx + 1}_AHU-{a_idx + 1}"
                btype = BUILDING_TYPES[b_idx % 4]
                segs = []
                for s_idx in range(self.segments_per_ahu):
                    fault = FAULT_CLASSES[s_idx % len(FAULT_CLASSES)]
                    fault_offset = 0.0 if fault == "normal" else 1.0 + 0.3 * FAULT_CLASSES.index(fault)
                    signal = synth_multichannel(
                        n_channels=len(CHANNELS), length=self.segment_length,
                        sampling_rate_hz=self.sampling_rate_hz, rng=rng,
                        base_freqs_hz=[1 / 86400, 1 / 86400, 1 / 86400, 1 / 86400,
                                       1 / 3600, 1 / 3600, 1 / 600, 1 / 600],
                        fault_offset=fault_offset, fault_channel=s_idx % len(CHANNELS),
                    )
                    segs.append(Segment(
                        segment_id=f"{bid}_seg{s_idx:03d}",
                        signal=signal, channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz, class_label=fault,
                        release_time=s_idx, timestamp_index=s_idx,
                    ))
                personas.append(Persona(
                    persona_id=bid, dataset=self.name,
                    identity_attributes={"building_type": btype, "ahu_model": "VAV-DDC"},
                    operating_attributes={"season": ["winter", "spring", "summer", "fall"][s_idx % 4],
                                          "occupancy_avg": round(float(rng.uniform(0.3, 0.95)), 2),
                                          "outdoor_temp_avg_c": round(float(rng.uniform(-5, 35)), 1)},
                    sensitive_attributes={"building_location": LOCATIONS[b_idx % 4],
                                          "owner": OWNERS[b_idx % 4],
                                          "occupant_count": int(rng.integers(50, 1500))},
                    segments=segs,
                ))
        return personas

    def _load_real(self) -> list[Persona]:
        # Real CSV parsing — minimal implementation
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for real LBNL parsing")

        personas = []
        ahu_dir = self.data_dir / "AHU"
        for bldg_dir in sorted(ahu_dir.iterdir()):
            if not bldg_dir.is_dir():
                continue
            csvs = list(bldg_dir.glob("*.csv"))[: self.segments_per_ahu]
            if not csvs:
                continue
            segs = []
            for i, csv_path in enumerate(csvs):
                try:
                    df = pd.read_csv(csv_path)
                    cols = [c for c in df.columns if c.lower() in [c.lower() for c in CHANNELS]]
                    if not cols:
                        cols = df.select_dtypes(include=[np.number]).columns.tolist()[: len(CHANNELS)]
                    sig = df[cols].iloc[: self.segment_length].T.to_numpy(dtype=float)
                    if sig.shape[1] < self.segment_length:
                        continue
                    if sig.shape[0] < len(CHANNELS):
                        sig = np.vstack([sig, np.zeros((len(CHANNELS) - sig.shape[0], sig.shape[1]))])
                    fault = "normal" if "baseline" in csv_path.stem.lower() else csv_path.stem
                    segs.append(Segment(
                        segment_id=f"{bldg_dir.name}_seg{i:03d}",
                        signal=sig[: len(CHANNELS)], channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz, class_label=fault,
                        release_time=i, timestamp_index=i,
                    ))
                except Exception as e:
                    logger.debug(f"Skip {csv_path}: {e}")
                    continue
            if segs:
                personas.append(Persona(
                    persona_id=bldg_dir.name, dataset=self.name,
                    identity_attributes={"building_type": "office", "ahu_model": "VAV-DDC"},
                    operating_attributes={"season": "summer", "occupancy_avg": 0.7,
                                          "outdoor_temp_avg_c": 20.0},
                    sensitive_attributes={"building_location": "real_unknown",
                                          "owner": "real_unknown", "occupant_count": 100},
                    segments=segs,
                ))
        if not personas:
            raise RuntimeError(f"No usable LBNL data under {self.data_dir}")
        return personas
