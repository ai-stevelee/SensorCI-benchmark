"""TEP (Tennessee Eastman Process) dataset loader.

Two modes:
  - synthetic (default): generates TEP-like multi-channel signals with embedded
    fault patterns; runs offline, useful for pilot.
  - real: downloads Rieth et al. extended TEP from Harvard Dataverse (requires
    internet + ~500MB).

Persona = simulation run; segments = windowed measurements.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)


# 41 measurement variables (XMEAS) + 12 manipulated variables (XMV)
# We use a simplified subset for the pilot: 5 representative channels
PILOT_CHANNELS = [
    "T_reactor",      # XMEAS(9) reactor temperature
    "P_reactor",      # XMEAS(7) reactor pressure
    "F_feed_C",       # XMEAS(4) feed flow C
    "T_sep",          # XMEAS(11) separator temperature
    "L_sep",          # XMEAS(12) separator level
]

# 21 fault types: IDV(0)=normal, IDV(1)..IDV(20)=faults
FAULT_NAMES = ["normal"] + [f"IDV{i}" for i in range(1, 21)]


class TEPDataset(BaseDatasetLoader):
    name = "tep"

    def __init__(
        self,
        mode: str = "synthetic",
        n_simulations: int = 12,
        segments_per_sim: int = 8,
        segment_length: int = 256,
        sampling_rate_hz: float = 1.0,
        seed: int = 42,
        **kw,
    ):
        super().__init__(**kw)
        self.mode = mode
        self.n_simulations = n_simulations
        self.segments_per_sim = segments_per_sim
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "synthetic":
            personas = self._load_synthetic()
        elif self.mode == "real":
            personas = self._load_real()
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")

        # Build attribute_categories then sharing matrix (stub heuristic;
        # replace via LLM-as-judge in Phase 3)
        attr_categories: dict[str, str] = {}
        for p in personas:
            for k in p.identity_attributes:
                attr_categories[k] = "identity"
            for k in p.operating_attributes:
                attr_categories[k] = "operating"
            for k in p.sensitive_attributes:
                attr_categories[k] = "sensitive"

        sharing = default_sharing_matrix(attr_categories)

        return TSDataset(
            name=self.name,
            personas=personas,
            sharing_matrix=sharing,
            class_list=FAULT_NAMES,
            dataset_metadata={
                "domain": "Chemical process simulation (Tennessee Eastman)",
                "citation": "Downs & Vogel 1993; Rieth et al. 2017",
                "channels": PILOT_CHANNELS,
                "sampling_rate_hz": self.sampling_rate_hz,
                "mode": self.mode,
                "n_personas": len(personas),
            },
        )

    # --------------------------------------------------------------------
    # Synthetic mode
    # --------------------------------------------------------------------

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas: list[Persona] = []
        # Spread fault types across simulations
        fault_assignments = (FAULT_NAMES * (self.n_simulations // len(FAULT_NAMES) + 1))[:self.n_simulations]
        rng.shuffle(fault_assignments)

        recipe_options = ["A1", "A2", "B1", "B2", "C1"]
        location_options = ["Plant_North", "Plant_South", "Plant_East"]

        for sim_idx in range(self.n_simulations):
            sim_id = f"TEP_run_{sim_idx:03d}"
            fault = fault_assignments[sim_idx]
            recipe = str(rng.choice(recipe_options))
            location = str(rng.choice(location_options))
            feed_rate = float(rng.uniform(0.7, 1.3))
            base_pressure = float(rng.uniform(2700, 2900))

            # Generate segments
            segs: list[Segment] = []
            for seg_idx in range(self.segments_per_sim):
                signal = self._generate_segment(
                    rng, fault, feed_rate, base_pressure, seg_idx,
                )
                segs.append(Segment(
                    segment_id=f"{sim_id}_seg{seg_idx:03d}",
                    signal=signal,
                    channels=PILOT_CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz,
                    class_label=fault,
                    release_time=seg_idx,
                    timestamp_index=seg_idx,
                ))

            personas.append(Persona(
                persona_id=sim_id,
                dataset=self.name,
                identity_attributes={
                    "recipe": recipe,
                    "plant_location": location,
                    "simulator_version": "TEP-2017-Rieth",
                },
                operating_attributes={
                    "feed_rate_relative": round(feed_rate, 3),
                    "base_pressure_kpa": round(base_pressure, 1),
                    "fault_class": fault,
                },
                sensitive_attributes={
                    "operator_company": str(rng.choice(["Eastman", "BASF", "Dow"])),
                    "campaign_id": f"campaign_{int(rng.integers(1, 50)):03d}",
                },
                segments=segs,
            ))
        return personas

    def _generate_segment(
        self,
        rng: np.random.Generator,
        fault: str,
        feed_rate: float,
        base_pressure: float,
        seg_idx: int,
    ) -> np.ndarray:
        """Generate one segment of shape (n_channels, segment_length).

        Each channel has a base trend + noise + (if fault != normal) a
        fault-specific perturbation. This is a *highly simplified* TEP
        analog; real Rieth dataset uses a full nonlinear simulator.
        """
        L = self.segment_length
        t = np.arange(L) / self.sampling_rate_hz
        n_ch = len(PILOT_CHANNELS)
        signal = np.zeros((n_ch, L))

        # Channel 0: T_reactor — base 120°C ± noise
        signal[0] = 120.0 + 2.0 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.5, L)
        # Channel 1: P_reactor — base_pressure ± noise
        signal[1] = base_pressure + 5.0 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 1.0, L)
        # Channel 2: F_feed_C — feed_rate ± noise
        signal[2] = feed_rate * 100 + rng.normal(0, 0.5, L)
        # Channel 3: T_sep — base 80°C ± noise
        signal[3] = 80.0 + 1.0 * np.sin(2 * np.pi * 0.005 * t) + rng.normal(0, 0.3, L)
        # Channel 4: L_sep — base 50% ± noise
        signal[4] = 50.0 + 0.5 * np.sin(2 * np.pi * 0.008 * t) + rng.normal(0, 0.2, L)

        # Fault perturbations (very simplified)
        if fault == "IDV1":  # A/C feed ratio
            signal[2] += 5 * np.tanh((t - L / 4) / 10)
        elif fault == "IDV4":  # reactor cooling water inlet
            signal[0] += 3.0 * (1 - np.exp(-t / 50))
        elif fault == "IDV5":  # condenser cooling water inlet
            signal[3] -= 2.0 * (1 - np.exp(-t / 30))
        elif fault.startswith("IDV") and fault != "normal":
            # Generic fault: add a small step or drift to a random channel
            ch = int(int(fault[3:]) % n_ch)
            signal[ch] += rng.normal(0, 0.5) * (t > L / 3).astype(float)

        return signal

    # --------------------------------------------------------------------
    # Real mode (Harvard Dataverse download stub)
    # --------------------------------------------------------------------

    def _load_real(self) -> list[Persona]:
        import requests
        rdata_path = self.data_dir / "TEP_FaultFree_Training.RData"

        if not rdata_path.exists():
            # Harvard Dataverse persistent URL
            url = ("https://dataverse.harvard.edu/api/access/datafile/"
                   ":persistentId?persistentId=doi:10.7910/DVN/6C3JR1/L4QQTM")
            logger.info(f"Downloading TEP data from {url} ...")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with rdata_path.open("wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            logger.info(f"Saved to {rdata_path} ({rdata_path.stat().st_size / 1e6:.1f} MB)")

        # TODO(P2.1-real): parse .RData with pyreadr or rpy2 and build personas
        # For now, fall back to synthetic with a note
        logger.warning(
            "Real TEP RData parser not yet implemented (TODO P2.1-real). "
            "Falling back to synthetic generator. To complete: install pyreadr and parse rdata_path."
        )
        return self._load_synthetic()
