import torch
from numpy.random import default_rng
import numpy as np
from collections import defaultdict
from typing import Iterable

neg_sampler_names = ['NegativeSampler', 'HeterogeneousNegativeSampler']

class NegativeSampler:
    def __init__(self, src_nodes: Iterable, dst_nodes: Iterable, dst_types: Iterable, name: str, seed: int = 9, 
                 check_link_existence: bool = True, strategy: str = 'random') -> None:
        
        self.neighs = defaultdict(set)
        if check_link_existence:
            for src, dst in zip(src_nodes, dst_nodes):
                if torch.is_tensor(src): src = src.item()
                if torch.is_tensor(dst): dst = dst.item()
                self.neighs[src].add(dst)

        self.src_nodes = src_nodes.unique().to('cpu')
        self.dst_nodes = dst_nodes.unique().to('cpu')
        self.seed = seed
        self.rng = default_rng(seed)
        self.name = name
        self.strategy = strategy
        self.check_link_existence = check_link_existence
        self.dst_nodes_np = self.dst_nodes.numpy()
        self._valid_neg_cache = {}

    def sample(self, src: torch.Tensor, dst_types: torch.Tensor, eval: bool = False, eval_seed: int = 9,  *args, **kwargs) -> torch.Tensor:
        rng = default_rng(eval_seed) if eval else self.rng
        src_np = src.detach().cpu().numpy()
        neg_dst = np.empty(src_np.shape[0], dtype=np.int64)

        if not self.check_link_existence:
            neg_dst = rng.choice(self.dst_nodes_np, size=src_np.shape[0])
        else:
            for src_id in np.unique(src_np):
                mask = src_np == src_id
                candidates = self._valid_neg_cache.get(src_id)
                if candidates is None:
                    seen = self.neighs[src_id]
                    candidates = np.array(
                        [dst for dst in self.dst_nodes_np if dst not in seen],
                        dtype=np.int64,
                    )
                    if candidates.size == 0:
                        candidates = self.dst_nodes_np
                    self._valid_neg_cache[src_id] = candidates
                neg_dst[mask] = rng.choice(candidates, size=mask.sum())

        return torch.tensor(neg_dst, dtype=torch.long, device=src.device)

    def _exists(self, src, dst):
        if torch.is_tensor(src): src = src.item()
        if torch.is_tensor(dst): dst = dst.item()        
        return dst in self.neighs[src]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'



class HeterogeneousNegativeSampler:
    def __init__(self, src_nodes: Iterable, dst_nodes: Iterable, dst_types: Iterable, name: str, seed: int = 9, 
                 check_link_existence: bool = True, strategy: str = 'random') -> None:
        
        self.neighs = defaultdict(set)

        # populating neighs
        if check_link_existence:
            for src, dst in zip(src_nodes, dst_nodes):
                if torch.is_tensor(src): src = src.item()
                if torch.is_tensor(dst): dst = dst.item()
                self.neighs[src].add(dst)

        self.src_nodes = src_nodes.unique().to('cpu')
        self.dst_nodes = {}
        for i in dst_types.unique():
            self.dst_nodes[i.item()] = dst_nodes[dst_types == i].unique().to('cpu')
        self.seed = seed
        self.rng = default_rng(seed)
        self.name = name
        self.strategy = strategy
        self.check_link_existence = check_link_existence
        self.dst_nodes_np = {k: v.numpy() for k, v in self.dst_nodes.items()}
        self._valid_neg_cache = {}

    def sample(self, src: torch.Tensor, dst_types: torch.Tensor, eval: bool = False, eval_seed: int = 9,  *args, **kwargs) -> torch.Tensor:
        rng = default_rng(eval_seed) if eval else self.rng
        src_np = src.detach().cpu().numpy()
        dst_types_np = dst_types.detach().cpu().numpy()
        neg_dst = np.empty(src_np.shape[0], dtype=np.int64)

        for src_id, dst_type in np.unique(np.stack([src_np, dst_types_np], axis=1), axis=0):
            mask = (src_np == src_id) & (dst_types_np == dst_type)
            dst_pool = self.dst_nodes_np[int(dst_type)]
            if not self.check_link_existence:
                candidates = dst_pool
            else:
                cache_key = (int(src_id), int(dst_type))
                candidates = self._valid_neg_cache.get(cache_key)
                if candidates is None:
                    seen = self.neighs[int(src_id)]
                    candidates = np.array(
                        [dst for dst in dst_pool if dst not in seen],
                        dtype=np.int64,
                    )
                    if candidates.size == 0:
                        candidates = dst_pool
                    self._valid_neg_cache[cache_key] = candidates
            neg_dst[mask] = rng.choice(candidates, size=mask.sum())

        return torch.tensor(neg_dst, dtype=torch.long, device=src.device)

    def _exists(self, src, dst):
        if torch.is_tensor(src): src = src.item()
        if torch.is_tensor(dst): dst = dst.item()
        return dst in self.neighs[src]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
