from sensorci.attacks.base import BaseAttack
from sensorci.attacks.a0_frontier import FrontierLLMAttack
from sensorci.attacks.a1_distinguish import DistinguishabilityAttack
from sensorci.attacks.a2_linear_recon import LinearReconstructionAttack
from sensorci.attacks.a3_neural_recon import NeuralReconstructionAttack
from sensorci.attacks.a4_ci_linkage import RoleStratifiedLinkageAttack
from sensorci.attacks.a5_stat_tests import StatisticalTestsAttack
from sensorci.attacks.a6_algebra import AlgebraProbeAttack
from sensorci.attacks.a7_mia_ts import MIATimeSeriesAttack

ATTACK_REGISTRY: dict[str, type[BaseAttack]] = {
    "a0_frontier": FrontierLLMAttack,
    "a1_distinguish": DistinguishabilityAttack,
    "a2_linear_recon": LinearReconstructionAttack,
    "a3_neural_recon": NeuralReconstructionAttack,
    "a4_ci_linkage": RoleStratifiedLinkageAttack,
    "a5_stat_tests": StatisticalTestsAttack,
    "a6_algebra": AlgebraProbeAttack,
    "a7_mia_ts": MIATimeSeriesAttack,
}


def get_attack(name: str, **kw) -> BaseAttack:
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack {name!r}. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**kw)


__all__ = [
    "BaseAttack",
    "FrontierLLMAttack", "DistinguishabilityAttack", "LinearReconstructionAttack",
    "NeuralReconstructionAttack", "RoleStratifiedLinkageAttack",
    "StatisticalTestsAttack", "AlgebraProbeAttack", "MIATimeSeriesAttack",
    "ATTACK_REGISTRY", "get_attack",
]
