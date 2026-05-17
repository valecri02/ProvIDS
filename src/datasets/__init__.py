import torch

from torch_geometric.datasets import JODIEDataset
from numpy.random import default_rng
import numpy
from .darpa import DARPADataset_Temporal, DARPADataset_Static
from .tgrab import TGRABDataset_Temporal


JODIE = ['Wikipedia', "Reddit", "MOOC", "LastFM"]

DARPA_PREFIXES = (
    "darpa_trace_",
    "darpa_theia_",
)
TGRAB_PREFIXES = (
    "darpa_tgrab_cause_effect",
    "darpa_tgrab_long_range",
    "darpa_tgrab_periodicity",
)


def is_darpa_dataset_name(name):
    return name.startswith(DARPA_PREFIXES)


def is_tgrab_dataset_name(name):
    return name.startswith(TGRAB_PREFIXES)


class DataNames(list):
    def __contains__(self, name):
        return (
            super().__contains__(name)
            or is_darpa_dataset_name(name)
            or is_tgrab_dataset_name(name)
        )


DATA_NAMES = DataNames(JODIE + list(DARPA_PREFIXES) + list(TGRAB_PREFIXES))

    
def get_dataset(root, name, version, seed, metadata=False):
    rng = default_rng(seed)
    data_metadata = ()
    if name in JODIE:
        dataset = JODIEDataset(root, name.lower())
        data = dataset[0]
        data.x = torch.tensor(rng.random((data.num_nodes,1), dtype=numpy.float32))
    elif is_tgrab_dataset_name(name):
        if version == 'temporal':
            dataset = TGRABDataset_Temporal(root, name)
            data = dataset[0]
        else:
            raise NotImplementedError
    elif is_darpa_dataset_name(name):
        if version == 'temporal':
            dataset = DARPADataset_Temporal(root, name)
            data = dataset[0]
            data_metadata = data.metadata
            del data.metadata
        elif version == 'static':
            dataset = DARPADataset_Static(root, name)
            data = dataset
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return data, data_metadata
