"""MIT-BIH Arrhythmia Database loader. wfdb (real) + synthetic fallback."""

from __future__ import annotations
import logging
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

CHANNELS = ["MLII", "V1"]
ARRHYTHMIA_CLASSES = ["normal", "PVC", "PAC", "AFib", "VTach",
                      "LBBB", "RBBB", "junctional"]
SEX_OPTIONS = ["M", "F"]


class MITBIHDataset(BaseDatasetLoader):
    name = "mitbih"

    def __init__(self, mode: str = "auto", n_records: int = 16, segments_per_record: int = 6,
                 segment_length: int = 1080,  # 3 sec at 360 Hz
                 sampling_rate_hz: float = 360.0, seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_records = n_records
        self.segments_per_record = segments_per_record
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real MIT-BIH load failed: {e}. Falling back to synthetic.")
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
            class_list=ARRHYTHMIA_CLASSES,
            dataset_metadata={
                "domain": "Cardiac ECG arrhythmia (medical)",
                "citation": "Goldberger et al., Circulation 2000",
                "channels": CHANNELS, "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return any(self.data_dir.glob("*.dat"))

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        record_ids = list(range(100, 100 + self.n_records))
        for r_id in record_ids:
            age = int(rng.integers(40, 90))
            sex = SEX_OPTIONS[(r_id % 2)]
            diag = ARRHYTHMIA_CLASSES[(r_id - 100) % len(ARRHYTHMIA_CLASSES)]

            segs = []
            for seg_idx in range(self.segments_per_record):
                # Each segment may have a different beat-pattern label
                beat_class = ARRHYTHMIA_CLASSES[seg_idx % len(ARRHYTHMIA_CLASSES)]
                fault_offset = ARRHYTHMIA_CLASSES.index(beat_class) * 0.2
                signal = synth_multichannel(
                    n_channels=len(CHANNELS), length=self.segment_length,
                    sampling_rate_hz=self.sampling_rate_hz, rng=rng,
                    base_freqs_hz=[1.2, 1.2],  # ~72 bpm
                    fault_offset=fault_offset, fault_channel=0, noise_std=0.1,
                )
                segs.append(Segment(
                    segment_id=f"MITBIH_{r_id}_seg{seg_idx:03d}",
                    signal=signal, channels=CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz, class_label=beat_class,
                    release_time=seg_idx, timestamp_index=seg_idx,
                ))

            personas.append(Persona(
                persona_id=f"MITBIH_{r_id}", dataset=self.name,
                identity_attributes={"record_id": str(r_id),
                                     "recording_setup": "MLII_V1_BIH_1979"},
                operating_attributes={"recording_duration_min": 30,
                                      "signal_quality": "high"},
                sensitive_attributes={"age": age, "sex": sex,
                                      "diagnosis": diag,
                                      "medications": "none_recorded"},
                segments=segs,
            ))
        return personas

    def _load_real(self) -> list[Persona]:
        try:
            import wfdb
        except ImportError:
            raise RuntimeError("wfdb required for real MIT-BIH parsing. pip install wfdb")

        personas = []
        # Find available records (.dat files)
        dat_files = sorted(self.data_dir.glob("*.dat"))[: self.n_records]
        for dat_path in dat_files:
            r_id = dat_path.stem
            try:
                rec = wfdb.rdrecord(str(self.data_dir / r_id),
                                    sampto=self.segment_length * self.segments_per_record)
                signals = rec.p_signal  # (n_samples, n_channels)
                if signals.shape[1] < 2:
                    continue
                signals = signals[:, :2].T  # (2, n_samples)
                segs = []
                for s_idx in range(self.segments_per_record):
                    start = s_idx * self.segment_length
                    end = start + self.segment_length
                    if end > signals.shape[1]:
                        break
                    sig = signals[:, start:end]
                    segs.append(Segment(
                        segment_id=f"MITBIH_{r_id}_seg{s_idx:03d}",
                        signal=sig, channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz,
                        class_label=ARRHYTHMIA_CLASSES[s_idx % len(ARRHYTHMIA_CLASSES)],
                        release_time=s_idx, timestamp_index=s_idx,
                    ))
                if segs:
                    personas.append(Persona(
                        persona_id=f"MITBIH_{r_id}", dataset=self.name,
                        identity_attributes={"record_id": r_id,
                                             "recording_setup": "MLII_V1_BIH_1979"},
                        operating_attributes={"recording_duration_min": 30,
                                              "signal_quality": "high"},
                        sensitive_attributes={"age": 65, "sex": "M",
                                              "diagnosis": "real_unknown",
                                              "medications": "real_unknown"},
                        segments=segs,
                    ))
            except Exception as e:
                logger.debug(f"Skip {dat_path}: {e}")
                continue
        if not personas:
            raise RuntimeError(f"No usable MIT-BIH records under {self.data_dir}")
        return personas
