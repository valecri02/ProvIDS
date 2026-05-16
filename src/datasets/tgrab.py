import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, TemporalData

#-------------------------------------------------------------------------------------------
# Code  copied from https://github.com/alirezadizaji/T-GRAB/blob/main/utils/types/node_feat.py

class NodeFeatType:
    CONSTANT = "CONSTANT"
    RAND = "RAND" # Sampling from uniform dist
    RANDN = "RANDN" # Sampling from normal dist
    ONE_HOT = "ONE_HOT"
    NODE_ID = "NODE_ID"

    @staticmethod
    def list():
        return [
            NodeFeatType.CONSTANT, 
            NodeFeatType.RAND,
            NodeFeatType.RANDN, 
            NodeFeatType.ONE_HOT, 
            NodeFeatType.NODE_ID
        ]

# Code  copied from https://github.com/alirezadizaji/T-GRAB/blob/main/utils/node_feat.py
class NodeFeatGenerator:
    def __init__(self, feat_type: str):
        self.feat_type = feat_type.upper()

    def __call__(self, num_nodes: int, node_feat_dim: int = 1) -> torch.Tensor:
        if self.feat_type == NodeFeatType.CONSTANT:
            node_feat = torch.ones((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.feat_type == NodeFeatType.RAND:
            node_feat = torch.rand((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.feat_type == NodeFeatType.RANDN:
            node_feat = torch.randn((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.feat_type == NodeFeatType.ONE_HOT:
            node_feat = torch.eye(num_nodes).float()
        elif self.feat_type == NodeFeatType.NODE_ID:
            node_feat = torch.arange(num_nodes).unsqueeze(1).float()
        else:
            raise ValueError(f"Unknown T-GRAB node feature type: {self.feat_type}")
        
        print(f"===========> Node feature generated: size {node_feat.size()}", flush=True)
        return node_feat
#-------------------------------------------------------------------------------------------


class TGRABDataset_Temporal(InMemoryDataset):
    """Adapter from T-GRAB data.npz files to ProvIDS TemporalData.

    Expected input layout:
        <root>/<name>/data.npz

    The produced object intentionally has DARPA-like fields (`ext_roll`,
    `hash_id`, `malicious`) so ProvIDS can reuse its existing temporal split
    and prediction code without changes to main.py or train_link.py.
    """

    def __init__(
        self,
        root: str,
        name: str,
        node_feat: str = NodeFeatType.NODE_ID,
        node_feat_dim: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        self.node_feat = node_feat
        self.node_feat_dim = node_feat_dim
        super().__init__(root, transform, pre_transform)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "temporal_processed")

    @property
    def raw_file_names(self) -> str:
        return "data.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        path = osp.join(self.root, self.name, "data.npz")
        data_np = np.load(path)
        data_np = {key: data_np[key] for key in data_np}

        required = [
            "src",
            "dst",
            "t",
            "edge_feat",
            "num_nodes",
            "train_mask",
            "val_mask",
            "test_mask",
            "test_inductive_mask",
        ]
        missing = [key for key in required if key not in data_np]
        if missing:
            raise ValueError(f"T-GRAB data file is missing keys: {missing}")

        src = torch.from_numpy(data_np["src"]).long()
        dst = torch.from_numpy(data_np["dst"]).long()
        t = torch.from_numpy(data_np["t"]).long()

        msg = torch.from_numpy(data_np["edge_feat"])
        if msg.ndim != 1:
            raise ValueError(f"Expected 1D T-GRAB edge features, got shape {tuple(msg.shape)}")
        msg = msg[:, None].float()

        num_nodes = int(data_np["num_nodes"])
        x = NodeFeatGenerator(self.node_feat)(num_nodes, self.node_feat_dim)

        # ProvIDS uses ext_roll as: 0=train, 1=validation, 2=test.
        # Merge T-GRAB inductive-test events into test for the unchanged
        # ProvIDS evaluation path.
        num_events = src.numel()
        ext_roll = torch.empty(num_events, dtype=torch.long)
        assigned = torch.zeros(num_events, dtype=torch.bool)

        split_specs = [
            ("train_mask", 0),
            ("val_mask", 1),
            ("test_mask", 2),
            ("test_inductive_mask", 2),
        ]
        for mask_key, split_value in split_specs:
            mask = torch.from_numpy(data_np[mask_key].astype(bool))
            ext_roll[mask] = split_value
            assigned |= mask

        if not bool(assigned.all()):
            raise ValueError("Some T-GRAB events were not assigned to any split")

        hash_id = torch.arange(num_events, dtype=torch.long)
        malicious = torch.zeros(num_events, dtype=torch.bool)

        data = TemporalData(
            src=src,
            dst=dst,
            t=t,
            msg=msg,
            x=x,
            ext_roll=ext_roll,
            hash_id=hash_id,
            malicious=malicious,
            is_tgrab=torch.tensor([True]),
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"
