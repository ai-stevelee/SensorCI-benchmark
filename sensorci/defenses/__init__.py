from sensorci.defenses.base import BaseDefense
from sensorci.defenses.d1_raw import RawDefense
from sensorci.defenses.d2_gaussian_dp import GaussianDPDefense
from sensorci.defenses.d3_laplace_dp import LaplaceDPDefense
from sensorci.defenses.d4_dp_ae import DPSGDAutoencoderDefense
from sensorci.defenses.d5_affine import PerSourceAffineDefense
from sensorci.defenses.d6_per_seg_affine import PerSegmentAffineDefense
from sensorci.defenses.d7_ope import OPEDefense
from sensorci.defenses.d8_doppelganger import DoppelGANgerDefense
from sensorci.defenses.d9_kanon import KAnonymityTSDefense
from sensorci.defenses.d10_fe_ip import FunctionalEncryptionIPDefense
from sensorci.defenses.d11_vfae import VFAETimeSeriesDefense
from sensorci.defenses.d12_ib import InformationBottleneckDefense

DEFENSE_REGISTRY: dict[str, type[BaseDefense]] = {
    "d1_raw": RawDefense,
    "d2_gaussian_dp": GaussianDPDefense,
    "d3_laplace_dp": LaplaceDPDefense,
    "d4_dp_ae": DPSGDAutoencoderDefense,
    "d5_affine": PerSourceAffineDefense,
    "d6_per_seg_affine": PerSegmentAffineDefense,
    "d7_ope": OPEDefense,
    "d8_doppelganger": DoppelGANgerDefense,
    "d9_kanon": KAnonymityTSDefense,
    "d10_fe_ip": FunctionalEncryptionIPDefense,
    "d11_vfae": VFAETimeSeriesDefense,
    "d12_ib": InformationBottleneckDefense,
}


def get_defense(name: str, **kw) -> BaseDefense:
    if name not in DEFENSE_REGISTRY:
        raise ValueError(f"Unknown defense {name!r}. Available: {list(DEFENSE_REGISTRY)}")
    return DEFENSE_REGISTRY[name](**kw)


__all__ = [
    "BaseDefense",
    "RawDefense", "GaussianDPDefense", "LaplaceDPDefense", "DPSGDAutoencoderDefense",
    "PerSourceAffineDefense", "PerSegmentAffineDefense", "OPEDefense",
    "DoppelGANgerDefense", "KAnonymityTSDefense",
    "FunctionalEncryptionIPDefense",
    "VFAETimeSeriesDefense", "InformationBottleneckDefense",
    "DEFENSE_REGISTRY", "get_defense",
]
