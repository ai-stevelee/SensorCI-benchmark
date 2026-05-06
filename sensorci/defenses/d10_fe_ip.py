"""D10: Functional Encryption for Inner Product (FE-IP).

Stylized utility-only proxy of Abdalla et al. PKC 2015. We do NOT implement the
actual cryptographic scheme (would need pairing-based crypto libs); instead, we
simulate its functional behavior:

  - Output: ciphertext-like opaque numbers (very low naturalness P_2)
  - Decoder: ONLY for inner-product queries f(x) = <w, x> (with key-sharing mode)
  - Computational cost: simulated as 1000x slower per query

Two evaluation modes (addresses Reviewer Q5: FE-IP key-sharing fairness):

  * mode="strict" (default): No decoder for queries outside <w, x> form.
    P_1 score is conservatively penalized → reflects real-world deployment
    where general-purpose analyst tools cannot use FE-IP output.

  * mode="key_sharing_analyst": Authorized analyst is given decryption keys
    for the cured query bank (limited to inner-product form). P_1 is
    measured for ONLY those queries the FE-IP scheme natively supports.
    This gives the optimistic upper bound for FE-IP utility.

Both modes are reported in the leaderboard for transparency.
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class FunctionalEncryptionIPDefense(BaseDefense):
    name = "d10_fe_ip"
    has_decoder = True  # only for inner-product queries

    def __init__(self, mode: str = "strict", security_param: int = 128, key_seed: int = 42):
        super().__init__()
        if mode not in ("strict", "key_sharing_analyst"):
            raise ValueError(f"mode must be 'strict' or 'key_sharing_analyst', got {mode!r}")
        self.mode = mode
        self.security_param = security_param
        self.key_seed = key_seed
        self._scaling: dict[str, float] = {}  # per-source ciphertext scaling

    def fit(self, dataset: TSDataset) -> None:
        rng = np.random.default_rng(self.key_seed)
        # Each source gets a per-component "key" (very stylized)
        for p in dataset.personas:
            self._scaling[p.persona_id] = float(rng.uniform(0.5, 2.0))

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        """FE-IP transform with two modes:

        * mode="strict" (default): cipher-like opaque output (per-source scaling +
          large Gaussian noise + per-channel time permutation). Models FE-IP as
          deployed without analyst keys -- destroys statistical naturalness AND
          time structure, so general-purpose query pipelines cannot use it.

        * mode="key_sharing_analyst": idealized FE-IP for keyed analyst.
          Per-source scaling + small calibrated Gaussian noise; NO time
          permutation (otherwise inner-product queries would be unrecoverable).
          The analyst possesses the per-source scaling key 's' and decodes
          inner-product queries via the matching decode_query.
        """
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        s = self._scaling.get(source_id, 1.0)
        if self.mode == "key_sharing_analyst":
            # Idealized keyed analyst: small calibrated noise, no permutation.
            noise_std = float(signal.std() * 0.1)
            return s * signal + rng.normal(0, noise_std, signal.shape)
        # Strict mode: cipher-like opaque output
        out = s * signal + rng.normal(0, signal.std() * 5.0, signal.shape)
        # Permute samples within each channel (destroys time structure)
        for c in range(out.shape[0]):
            perm = rng.permutation(out.shape[1])
            out[c] = out[c, perm]
        return out

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        """Decode an inner-product query, ONLY in key_sharing mode AND for inner-product queries."""
        if self.mode != "key_sharing_analyst":
            return y_query  # Strict mode: no decoder available
        if query_meta.get("kind") not in ("inner_product", "channel_stat"):
            return y_query  # Non-IP query: not supported
        # For supported query, "decrypt" by undoing scaling
        s = self._scaling.get(source_id, 1.0)
        if s == 0:
            return y_query
        return y_query / s

    @property
    def hyperparameters(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "security_param": self.security_param,
            "key_seed": self.key_seed,
        }
