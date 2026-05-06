# Extending SensorCI

Adding a new dataset or defense to SensorCI is intentionally narrow: implement one base class and register it in the pipeline.

## Adding a new dataset

1. Implement `sensorci/datasets/<your_dataset>.py` subclassing `core.dataset.TSDataset`:

```python
from sensorci.core.dataset import TSDataset
from sensorci.core.persona import Persona, Segment

class MyDataset(TSDataset):
    name = "my_dataset"

    def load(self, data_dir):
        # populate self.personas: list[Persona]
        # each Persona has .persona_id, .all_attributes (dict), .segments (list[Segment])
        # each Segment has .segment_id, .signal (np.ndarray, shape [T, C])
        ...

    def attribute_set(self):
        # return set of attribute names
        ...

    def not_share_attributes(self, role):
        # return list of attributes Phi_D[role, attr] == 'N'
        # OR call self.sharing_matrix.not_share(role) if curated via judge
        ...
```

2. Register in `sensorci/datasets/__init__.py`:
```python
from .my_dataset import MyDataset
DATASET_REGISTRY["my_dataset"] = MyDataset
```

3. Run the LLM-as-Judge curation pipeline to generate Φ_D:
```bash
python scripts/curation/20_judge_query_bank.py --dataset my_dataset
python scripts/curation/21_judge_role_matrix.py --dataset my_dataset
```

4. Smoke-test with a single defense:
```bash
python scripts/main_eval/30_main_evaluation.py \
    --datasets my_dataset --defenses d1_raw --attacks a1_distinguish \
    --tiers T1 --seeds 42
```

## Adding a new defense

1. Implement `sensorci/defenses/d_yourname.py` subclassing `defenses.base.BaseDefense`:

```python
from sensorci.defenses.base import BaseDefense

class DYourName(BaseDefense):
    name = "d_yourname"

    def __init__(self, key_seed: int = 42, hyperparam: float = 0.1):
        self.key_seed = key_seed
        self.hyperparam = hyperparam

    def fit(self, dataset):
        # optional pre-fit on the train split (for D4_dp_ae, D8_doppelganger, D11_vfae, D12_ib)
        ...

    def transform(self, signal, source_id, segment_id):
        # apply the privacy transformation T_theta
        # source_id and segment_id are passed for per-source / per-segment keying
        return transformed_signal
```

2. Register in `sensorci/defenses/__init__.py`:
```python
DEFENSE_REGISTRY["d_yourname"] = DYourName
```

3. Add hyperparameter grid to `configs/paper_full.yaml`:
```yaml
d_yourname:
  hyperparam: [0.01, 0.1, 1.0]
```

4. Run the val-split sweep to select hyperparameters:
```bash
python scripts/main_eval/30_main_evaluation.py \
    --defenses d_yourname --val_split --out_dir results/sweep_d_yourname/
```

The selected setting is automatically picked when you run the test split with `--use_val_selected`.

## Adding a new attack

Subclass `sensorci/attacks/base.BaseAttack`. Implement `run(defense, dataset, n_samples, seed) -> AttackResult`. Register in `sensorci/attacks/__init__.py`.

## Smoke-testing changes

```bash
pytest tests/ -v
```

Test directory ships unit tests for each registered (defense, attack, dataset) combination on small synthetic data.
