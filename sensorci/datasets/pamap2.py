"""PAMAP2 (Physical Activity Monitoring) loader. Real .dat + synthetic fallback."""

from __future__ import annotations
import logging
import numpy as np

from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets._synthetic import synth_multichannel
from sensorci.core.persona import Persona, Segment
from sensorci.core.dataset import TSDataset, default_sharing_matrix

logger = logging.getLogger(__name__)

# 3 IMUs (chest, hand, ankle) * (3 acc + 3 gyro + 3 mag) + heart_rate
CHANNELS = (
    [f"chest_acc_{a}" for a in "xyz"]
    + [f"chest_gyro_{a}" for a in "xyz"]
    + [f"chest_mag_{a}" for a in "xyz"]
    + [f"hand_acc_{a}" for a in "xyz"]
    + [f"hand_gyro_{a}" for a in "xyz"]
    + [f"ankle_acc_{a}" for a in "xyz"]
    + ["heart_rate"]
)
ACTIVITIES = [
    "lying", "sitting", "standing", "walking", "running", "cycling",
    "Nordic_walking", "ascending_stairs", "descending_stairs",
    "vacuum_cleaning", "ironing", "rope_jumping",
]
SEX_OPTIONS = ["M", "F"]
FITNESS_OPTIONS = ["high", "medium", "low"]


class PAMAP2Dataset(BaseDatasetLoader):
    name = "pamap2"

    def __init__(self, mode: str = "auto", n_subjects: int = 9, segments_per_subject: int = 6,
                 segment_length: int = 256, sampling_rate_hz: float = 100.0,
                 seed: int = 42, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.n_subjects = n_subjects
        self.segments_per_subject = segments_per_subject
        self.segment_length = segment_length
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed

    def load(self) -> TSDataset:
        if self.mode == "real" or (self.mode == "auto" and self._has_real_data()):
            try:
                personas = self._load_real()
            except Exception as e:
                logger.warning(f"Real PAMAP2 load failed: {e}. Falling back to synthetic.")
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
            class_list=ACTIVITIES,
            dataset_metadata={
                "domain": "Human activity recognition (HAR) — IMU + heart rate",
                "citation": "Reiss & Stricker, ISWC 2012",
                "channels": CHANNELS, "sampling_rate_hz": self.sampling_rate_hz,
                "mode": "real" if self._has_real_data() else "synthetic",
            },
        )

    def _has_real_data(self) -> bool:
        return (self.data_dir / "Protocol" / "subject101.dat").exists()

    def _load_synthetic(self) -> list[Persona]:
        rng = np.random.default_rng(self.seed)
        personas = []
        for s_idx in range(self.n_subjects):
            sub_id = 101 + s_idx
            age = int(rng.integers(20, 50))
            sex = SEX_OPTIONS[s_idx % 2]
            bmi = round(float(rng.uniform(18, 30)), 1)
            fitness = FITNESS_OPTIONS[s_idx % 3]

            segs = []
            for seg_idx in range(self.segments_per_subject):
                activity = ACTIVITIES[seg_idx % len(ACTIVITIES)]
                # Encode activity in fault_offset (just a way to distinguish classes)
                fault_offset = ACTIVITIES.index(activity) * 0.3
                signal = synth_multichannel(
                    n_channels=len(CHANNELS), length=self.segment_length,
                    sampling_rate_hz=self.sampling_rate_hz, rng=rng,
                    fault_offset=fault_offset, fault_channel=0,
                )
                # Heart rate channel: mean ~70 + activity-dependent
                signal[-1] = 70 + ACTIVITIES.index(activity) * 5 + rng.normal(0, 3, self.segment_length)
                segs.append(Segment(
                    segment_id=f"subject{sub_id}_seg{seg_idx:03d}",
                    signal=signal, channels=CHANNELS,
                    sampling_rate_hz=self.sampling_rate_hz, class_label=activity,
                    release_time=seg_idx, timestamp_index=seg_idx,
                ))

            personas.append(Persona(
                persona_id=f"PAMAP2_subject_{sub_id}", dataset=self.name,
                identity_attributes={"subject_id": sub_id, "dominant_hand": "right"},
                operating_attributes={"protocol": "Protocol",
                                      "activity_intensity": "mixed"},
                sensitive_attributes={"age": age, "sex": sex,
                                      "bmi": bmi, "fitness_level": fitness},
                segments=segs,
            ))
        return personas

    def _load_real(self) -> list[Persona]:
        protocol_dir = self.data_dir / "Protocol"
        if not protocol_dir.exists():
            raise RuntimeError("No Protocol/ directory in PAMAP2 data")

        # PAMAP2 column layout (0-indexed):
        # 0: timestamp, 1: activity_id, 2: heart_rate
        # 3: chest_temp, 4-6: chest_acc16(xyz), 7-9: chest_acc6(xyz),
        # 10-12: chest_gyro(xyz), 13-15: chest_mag(xyz), 16-19: chest_orient(4)
        # 20: hand_temp, 21-23: hand_acc16(xyz), 24-26: hand_acc6(xyz),
        # 27-29: hand_gyro(xyz), 30-32: hand_mag(xyz), 33-36: hand_orient(4)
        # 37: ankle_temp, 38-40: ankle_acc16(xyz), 41-43: ankle_acc6(xyz),
        # 44-46: ankle_gyro(xyz), 47-49: ankle_mag(xyz), 50-53: ankle_orient(4)
        CHANNEL_COLS = [
            4, 5, 6,      # chest_acc x,y,z
            10, 11, 12,   # chest_gyro x,y,z
            13, 14, 15,   # chest_mag x,y,z
            21, 22, 23,   # hand_acc x,y,z
            27, 28, 29,   # hand_gyro x,y,z
            38, 39, 40,   # ankle_acc x,y,z
            2,            # heart_rate
        ]

        personas = []
        for dat_path in sorted(protocol_dir.glob("subject*.dat"))[: self.n_subjects]:
            try:
                data = np.genfromtxt(dat_path, filling_values=np.nan)
                if data.ndim != 2 or data.shape[1] < 54:
                    continue

                # Forward-fill NaN per column
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    mask = np.isnan(col_data)
                    if mask.any():
                        valid_idx = np.where(~mask)[0]
                        if len(valid_idx) == 0:
                            data[:, col] = 0.0
                        else:
                            data[mask, col] = np.interp(
                                np.where(mask)[0], valid_idx, col_data[valid_idx]
                            )

                segs = []
                for s_idx in range(self.segments_per_subject):
                    start = s_idx * self.segment_length
                    end = start + self.segment_length
                    if end > len(data):
                        break
                    sig = data[start:end, CHANNEL_COLS].T.astype(np.float32)
                    activity_id = int(data[start, 1]) if not np.isnan(data[start, 1]) else 0
                    activity = ACTIVITIES[min(activity_id - 1, len(ACTIVITIES) - 1)] if activity_id > 0 else "lying"
                    segs.append(Segment(
                        segment_id=f"{dat_path.stem}_seg{s_idx:03d}",
                        signal=sig, channels=CHANNELS,
                        sampling_rate_hz=self.sampling_rate_hz, class_label=activity,
                        release_time=s_idx, timestamp_index=s_idx,
                    ))
                if segs:
                    sub_id = int(dat_path.stem.replace("subject", ""))
                    personas.append(Persona(
                        persona_id=f"PAMAP2_subject_{sub_id}", dataset=self.name,
                        identity_attributes={"subject_id": sub_id, "dominant_hand": "right"},
                        operating_attributes={"protocol": "Protocol", "activity_intensity": "mixed"},
                        sensitive_attributes={"age": 30, "sex": "M", "bmi": 22.0, "fitness_level": "medium"},
                        segments=segs,
                    ))
            except Exception as e:
                logger.debug(f"Skip {dat_path}: {e}")
                continue
        if not personas:
            raise RuntimeError(f"No usable PAMAP2 subjects under {self.data_dir}")
        return personas
