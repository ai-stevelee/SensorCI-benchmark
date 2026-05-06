"""NASA C-MAPSS Turbofan loader. Real (txt) + synthetic fallback."""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

CHANNELS = [f"sensor_{i}" for i in range(1, 22)] + ["op_setting_1", "op_setting_2", "op_setting_3"]
FAULT_CLASSES = ["healthy_early", "healthy_mid", "degrading_early", "degrading_late", "near_failure"]
FD_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ENGINE_FAMILIES = ["commercial-turbofan", "narrow-body", "wide-body"]
OPERATORS = ["Airline_A", "Airline_B", "Airline_C"]


class CMAPSSDataset(BaseDatasetLoader):
    name = "cmapss"

    def __init__(self, mode: str = "auto", n_engines: int = 24, segments_per_engine: int = 4,
                 segment_length: int = 128, sampling_rate_hz: float = 1.0,
                 seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_engines = n_engines
        self.segments_per_engine = segments_per_engine
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real C-MAPSS load failed: {e}. Falling back to synthetic.")
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
                "domain": "Aircraft turbofan engine simulation (RUL prediction)",
                "citation": "Saxena et al., PHM 2008",
                "channels": CHANNELS, "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return (self.data_dir / "train_FD001.txt").exists()

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        for e_idx in range(self.n_engines):
            fd = FD_SUBSETS[e_idx % len(FD_SUBSETS)]
            family = ENGINE_FAMILIES[e_idx % 3]
            operator = OPERATORS[e_idx % 3]
            altitude = float(rng.uniform(20000, 40000))
            mach = float(rng.uniform(0.7, 0.9))

            segs = []
            for s_idx in range(self.segments_per_engine):
                fault_class = FAULT_CLASSES[min(s_idx, len(FAULT_CLASSES) - 1)]
                fault_offset = 0.2 * s_idx
                signal = synth_multichannel(
                    n_channels=len(CHANNELS), length=self.segment_length,
                    sampling_rate_hz=self.sampling_rate_hz, rng=rng,
                    fault_offset=fault_offset, fault_channel=0,
                )
                segs.append(Segment(
                    segment_id=f"{fd}_unit_{e_idx:03d}_seg{s_idx:03d}",
                    signal=signal, channels=CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz, class_label=fault_class,
                    release_time=s_idx, timestamp_index=s_idx,
                ))

            personas.append(Persona(
                persona_id=f"{fd}_unit_{e_idx:03d}", dataset=self.name,
                identity_attributes={"fd_subset": fd, "engine_family": family},
                operating_attributes={"altitude_ft": round(altitude, 0),
                                      "mach_number": round(mach, 3),
                                      "throttle_resolver_angle": float(rng.uniform(60, 100))},
                sensitive_attributes={"operator": operator,
                                      "maintenance_history_id": f"mhx_{e_idx:03d}",
                                      "fleet_age_years": int(rng.integers(2, 15))},
                segments=segs,
            ))
        return personas

    def _load_real(self) -> list[Persona]:
        personas = []
        for fd in FD_SUBSETS:
            train_path = self.data_dir / f"train_{fd}.txt"
            if not train_path.exists():
                continue
            data = np.loadtxt(train_path)
            unit_ids = np.unique(data[:, 0])
            for uid in unit_ids[: self.n_engines // len(FD_SUBSETS) + 1]:
                rows = data[data[:, 0] == uid]
                if len(rows) < self.segment_length:
                    continue
                segs = []
                for s_idx in range(self.segments_per_engine):
                    start = s_idx * self.segment_length
                    end = start + self.segment_length
                    if end > len(rows):
                        break
                    # Channels 5: are sensors + op settings
                    sig = rows[start:end, 2:].T  # (n_channels, segment_length)
                    if sig.shape[0] < len(CHANNELS):
                        pad = np.zeros((len(CHANNELS) - sig.shape[0], sig.shape[1]))
                        sig = np.vstack([sig, pad])
                    rul = max(0, len(rows) - end)
                    fault_class = FAULT_CLASSES[min(int(rul / max(1, len(rows) // 5)), 4)]
                    segs.append(Segment(
                        segment_id=f"{fd}_unit_{int(uid):03d}_seg{s_idx:03d}",
                        signal=sig[: len(CHANNELS)], channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz, class_label=fault_class,
                        release_time=s_idx, timestamp_index=s_idx,
                    ))
                if segs:
                    personas.append(Persona(
                        persona_id=f"{fd}_unit_{int(uid):03d}", dataset=self.name,
                        identity_attributes={"fd_subset": fd, "engine_family": "commercial-turbofan"},
                        operating_attributes={"altitude_ft": float(rows[0, 2]),
                                              "mach_number": float(rows[0, 3]),
                                              "throttle_resolver_angle": float(rows[0, 4])},
                        sensitive_attributes={"operator": "real_unknown",
                                              "maintenance_history_id": f"mhx_{int(uid):03d}",
                                              "fleet_age_years": 5},
                        segments=segs,
                    ))
            if len(personas) >= self.n_engines:
                break
        if not personas:
            raise RuntimeError(f"No usable C-MAPSS data under {self.data_dir}")
        return personas
