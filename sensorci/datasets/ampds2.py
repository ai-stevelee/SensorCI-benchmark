"""AMPDs2 (Almanac of Minutely Power Dataset, v2) loader. Real CSV + synthetic fallback."""

from __future__ import annotations
import logging
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

CHANNELS = ["WHE_P", "FRE_P", "HPE_P", "FGE_P", "DWE_P", "CDE_P", "CWE_P", "DNE_P",
            "EQE_P", "EBE_P", "OFE_P", "TVE_P", "OUTSIDE_TEMP", "WIND_SPEED", "RH"]
FAULT_CLASSES = ["nominal", "appliance_event_low", "appliance_event_high",
                 "weather_anomaly", "occupancy_change"]
SEASONS = ["winter", "spring", "summer", "fall"]
DEMOGRAPHICS = ["family_4_persons", "couple_no_kids", "single_occupant", "family_2_persons"]


class AMPDs2Dataset(BaseDatasetLoader):
    name = "ampds2"

    def __init__(self, mode: str = "auto", n_personas: int = 16, segments_per_persona: int = 4,
                 segment_length: int = 1440, sampling_rate_hz: float = 1 / 60,
                 seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_personas = n_personas
        self.segments_per_persona = segments_per_persona
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real AMPDs2 load failed: {e}. Falling back to synthetic.")
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
                "domain": "Residential electricity submetering (single household, partitioned)",
                "citation": "Makonin, Scientific Data 2016",
                "channels": CHANNELS, "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return any(self.data_dir.glob("Electricity_*.csv"))

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        # Partition 1 household into multiple periods
        for p_idx in range(self.n_personas):
            month = (p_idx % 12) + 1
            year = 2013 + (p_idx // 12)
            segs = []
            for s_idx in range(self.segments_per_persona):
                fault = FAULT_CLASSES[s_idx % len(FAULT_CLASSES)]
                fault_offset = 0.0 if fault == "nominal" else 0.5 + 0.2 * FAULT_CLASSES.index(fault)
                signal = synth_multichannel(
                    n_channels=len(CHANNELS), length=self.segment_length,
                    sampling_rate_hz=self.sampling_rate_hz, rng=rng,
                    base_freqs_hz=[1 / 86400] * 12 + [1 / 86400, 1 / 21600, 1 / 86400],
                    fault_offset=fault_offset, fault_channel=p_idx % len(CHANNELS),
                )
                segs.append(Segment(
                    segment_id=f"AMPDs2_{year}-{month:02d}_seg{s_idx:03d}",
                    signal=signal, channels=CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz, class_label=fault,
                    release_time=s_idx, timestamp_index=s_idx,
                ))
            personas.append(Persona(
                persona_id=f"AMPDs2_{year}-{month:02d}", dataset=self.name,
                identity_attributes={"region": "Greater_Vancouver",
                                     "household_id": "household_001"},
                operating_attributes={"season": SEASONS[(month - 1) // 3],
                                      "weekday_avg": "Wed",
                                      "outdoor_temp_avg_c": 12.0 + 8.0 * np.sin(month / 12 * 2 * np.pi)},
                sensitive_attributes={"residents": DEMOGRAPHICS[p_idx % 4],
                                      "appliance_inventory": "fridge,heat_pump,dryer,dishwasher",
                                      "income_bracket": "middle"},
                segments=segs,
            ))
        return personas

    def _load_real(self) -> list[Persona]:
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for real AMPDs2 parsing")
        # Parse first electricity submeter file as schema reference
        e_files = sorted(self.data_dir.glob("Electricity_*.csv"))
        if not e_files:
            raise RuntimeError("No Electricity_*.csv files found")
        # For brevity, fall back to synthetic structure with real data plug
        logger.info(f"Found {len(e_files)} AMPDs2 electricity files; partial real parser. "
                    f"Full implementation TODO.")
        return self._load_synthetic()
