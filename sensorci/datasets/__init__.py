from sensorci.datasets.base import BaseDatasetLoader
from sensorci.datasets.tep import TEPDataset
from sensorci.datasets.paderborn import PaderbornDataset
from sensorci.datasets.cmapss import CMAPSSDataset
from sensorci.datasets.lbnl_fdd import LBNLFDDDataset
from sensorci.datasets.ampds2 import AMPDs2Dataset
from sensorci.datasets.pamap2 import PAMAP2Dataset
from sensorci.datasets.mitbih import MITBIHDataset

LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "tep": TEPDataset,
    "paderborn": PaderbornDataset,
    "cmapss": CMAPSSDataset,
    "lbnl_fdd": LBNLFDDDataset,
    "ampds2": AMPDs2Dataset,
    "pamap2": PAMAP2Dataset,
    "mitbih": MITBIHDataset,
}


def load_dataset(name: str, **kw):
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[name](**kw).load()


__all__ = ["BaseDatasetLoader", "TEPDataset", "PaderbornDataset", "CMAPSSDataset",
           "LBNLFDDDataset", "AMPDs2Dataset", "PAMAP2Dataset", "MITBIHDataset",
           "LOADER_REGISTRY", "load_dataset"]
