import torch
from typing import Callable, Optional, Dict, Tuple
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import TGNMemory 
from torch import Tensor
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.utils import scatter

TGNMessageStoreType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
TGNMessageStoreWithZType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class IdentityLayer(torch.nn.Module):
    # NOTE: this object is used to implement those models that do not have a RNN-based memory
    def __init__(self):
          super().__init__()
          self.I = torch.nn.Identity()
    
    def forward(self, x, *args, **kwargs):
         return self.I(x)
    

class NoMemory(torch.nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int, time_dim:int, init_time: int = 0) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_enc = TimeEncoder(time_dim)

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor,
                     raw_msg: Tensor):
        n_id = torch.cat([src, dst]).unique()
        self.last_update[n_id] = t.max()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id):
        return self.memory[n_id], self.last_update[n_id]


class GeneralMemory(TGNMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable,
                 rnn: Optional[str] = None,
                 non_linearity: str = 'tanh',
                 init_time: int = 0,
                 message_batch: int = 200):
        
        super().__init__(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)

        self.message_batch = message_batch
        if rnn is None:
             self.gru = IdentityLayer()
        else:
            rnn_instance = getattr(torch.nn, rnn)
            if 'RNN' in rnn:
                self.gru = rnn_instance(message_module.out_channels, memory_dim, nonlinearity=non_linearity)
            else:
                self.gru = rnn_instance(message_module.out_channels, memory_dim)

        self.memory[:] = torch.zeros(num_nodes, memory_dim).type_as(self.memory)
        self.last_update[:] = torch.ones(num_nodes).type_as(self.last_update) * init_time

        if hasattr(self.gru, 'reset_parameters'):
            self.gru.reset_parameters()

        # Store mode: 'base' uses PyG's internal message stores, 'z' uses
        # GNN-provided embeddings (z_src/z_dst) for message construction.
        self._store_mode: str = 'base'
        self._reset_message_store_z()

    def _reset_message_store_z(self):
        i = self.memory.new_empty((0, ), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        z = self.memory.new_empty((0, self.memory.size(-1)), device=self.device)
        # Message store format: (src, dst, t, msg, z_src, z_dst)
        self.msg_s_store_z: TGNMessageStoreWithZType = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}
        self.msg_d_store_z: TGNMessageStoreWithZType = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}

    def reset_state(self):
        super().reset_state()
        self._reset_message_store_z()

    def _update_msg_store_z(self, src: Tensor, dst: Tensor, t: Tensor,
                            raw_msg: Tensor, z_src: Tensor, z_dst: Tensor,
                            msg_store: TGNMessageStoreWithZType):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx], z_src[idx], z_dst[idx])

    def _compute_msg_z(self, n_id: Tensor, msg_store: TGNMessageStoreWithZType,
                       msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg, z_src, z_dst = list(zip(*data))

        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)

        # Filter out empty tensors to avoid `invalid configuration argument`.
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        z_src = [m for i, m in enumerate(z_src) if m.numel() > 0 or i == 0]
        z_dst = [m for i, m in enumerate(z_dst) if m.numel() > 0 or i == 0]

        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        z_src = torch.cat(z_src, dim=0).to(self.device)
        z_dst = torch.cat(z_dst, dim=0).to(self.device)

        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(z_src, z_dst, raw_msg, t_enc)
        return msg, t, src, dst

    def _get_updated_memory(self, n_id: Tensor):
        if getattr(self, '_store_mode', 'base') != 'z':
            return super()._get_updated_memory(n_id)

        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst) using stored GNN embeddings.
        msg_s, t_s, src_s, dst_s = self._compute_msg_z(n_id, self.msg_s_store_z,
                                                       self.msg_s_module)
        # Compute messages (dst -> src) using stored GNN embeddings.
        msg_d, t_d, src_d, dst_d = self._compute_msg_z(n_id, self.msg_d_store_z,
                                                       self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]

        return memory, last_update

    def update_state_with_z(self, src: Tensor, dst: Tensor, t: Tensor,
                            raw_msg: Tensor, z_src: Tensor, z_dst: Tensor):
        """Updates the memory using externally provided node embeddings.

        This mirrors :meth:`torch_geometric.nn.TGNMemory.update_state`, but
        stores `(z_src, z_dst)` per interaction and uses them for message
        construction instead of `self.memory[src]`/`self.memory[dst]`.
        """
        self._store_mode = 'z'

        if z_src.size(0) != raw_msg.size(0) or z_dst.size(0) != raw_msg.size(0):
            raise ValueError(
                f"update_state_with_z: expected z_src/z_dst batch={raw_msg.size(0)} "
                f"but got z_src={z_src.size(0)} z_dst={z_dst.size(0)}")
        mem_dim = self.memory.size(-1)
        if z_src.size(-1) != mem_dim or z_dst.size(-1) != mem_dim:
            raise ValueError(
                f"update_state_with_z: expected embedding dim={mem_dim} "
                f"but got z_src={z_src.size(-1)} z_dst={z_dst.size(-1)}")

        self._last_updated_n_id = torch.cat([src, dst]).unique()
        n_id = self._last_updated_n_id

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            # Reverse direction store: (dst -> src) with swapped embeddings.
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
        else:
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
            self._update_memory(n_id)

    # ------ TGNMemory methods ------------------------------------------------------------------------
    # self.reset_parameters() -- Resets all learnable parameters of the module. 
    # self.reset_state() -- Resets the memory to its initial state.
    # self.detach() -- Detaches the memory from gradient computation.
    # self.forward(n_id: Tensor) -- 
    #   Returns, for all nodes :obj:`n_id`, their current memory and their last updated timestamp.
    # self.update_state(src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor)
    #   Updates the memory with newly encountered interactions: obj:`(src, dst, t, raw_msg)`.
    # -------------------------------------------------------------------------------------------------
    
    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # Do it in batches of nodes, otherwise CUDA runs out of memory for datasets with millions of nodes
            for i in range(0, self.num_nodes, self.message_batch):
                self._update_memory(
                    torch.arange(i, min(self.num_nodes, i + self.message_batch), device=self.memory.device))
            self._reset_message_store()
            self._reset_message_store_z()
        super(TGNMemory, self).train(mode)

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        # Track nodes touched in the last update so we can efficiently detach
        # only the relevant message-store entries after the optimizer step.
        self._store_mode = 'base'
        self._last_updated_n_id = torch.cat([src, dst]).unique()
        return super().update_state(src, dst, t, raw_msg)

    def detach(self):
        """Detaches the memory *and* stored messages from gradient computation.

        PyG's :class:`~torch_geometric.nn.TGNMemory` only detaches `self.memory`.
        However, the per-node message stores (`msg_s_store`, `msg_d_store`) may
        contain tensors that keep the autograd graph alive across batches.
        """
        super().detach()

        n_id = getattr(self, '_last_updated_n_id', None)
        if n_id is None:
            return

        node_ids = n_id.tolist() if torch.is_tensor(n_id) else list(n_id)
        for store_name in ('msg_s_store', 'msg_d_store'):
            store = getattr(self, store_name, None)
            if store is None:
                continue
            for i in node_ids:
                src_i, dst_i, t_i, raw_msg_i = store[i]
                store[i] = (src_i, dst_i, t_i, raw_msg_i.detach())

        for store_name in ('msg_s_store_z', 'msg_d_store_z'):
            store = getattr(self, store_name, None)
            if store is None:
                continue
            for i in node_ids:
                src_i, dst_i, t_i, raw_msg_i, z_src_i, z_dst_i = store[i]
                store[i] = (src_i, dst_i, t_i, raw_msg_i.detach(), z_src_i.detach(), z_dst_i.detach())
        


class DyRepMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + memory_dim + time_dim

    def forward(self, z_dst, raw_msg, t_enc):
        return torch.cat([z_dst, raw_msg, t_enc], dim=-1)
    
class DyRepMemory(GeneralMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 message_module: Callable,
                 aggregator_module: Callable,
                 non_linearity: str = 'tanh',
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
                ):

        super().__init__(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=memory_dim, time_dim=1, 
                         message_module=message_module, aggregator_module=aggregator_module, rnn='RNNCell', 
                         non_linearity=non_linearity, init_time=init_time)
        self.conv = TransformerConv(memory_dim, memory_dim, edge_dim=raw_msg_dim,
                                    root_weight=False, aggr='max')
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        
        self.register_buffer('_mapper', torch.empty(num_nodes,
                                                   dtype=torch.long))
        
        if hasattr(self.conv, 'reset_parameters'):
            self.conv.reset_parameters()

    def _compute_msg(self, n_id: torch.Tensor, msg_store: TGNMessageStoreType,
                     msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)

        node_id = torch.cat([src, dst], dim=0).unique()
        self._mapper[node_id] = torch.arange(node_id.size(0), device=n_id.device)
        edge_index = torch.stack((self._mapper[src], self._mapper[dst])).long()
        x = self.memory[node_id]

        h_struct = self.conv(x, edge_index, edge_attr=raw_msg)

        t_rel = (t - self.last_update[src]).view(-1, 1)
        t_rel = (t_rel - self.mean_delta_t) / self.std_delta_t # delta_t normalization

        msg = msg_module(h_struct[self._mapper[dst]], raw_msg, t_rel)

        return msg, t, src, dst


class SimpleMemory(torch.nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int, init_time: int = 0) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)

    def update(self, n_id, new_mem_values, last_update):
        self.memory[n_id] = new_mem_values
        self.last_update[n_id] = last_update

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id):
        return self.memory[n_id], self.last_update[n_id]
    

class LastUpdateMemory(torch.nn.Module):
    def __init__(self, num_nodes: int, init_time: int = 0) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)

    def update_state(self, src, pos_dst, t, *args, **kwargs):
        self.last_update[src] = t
        self.last_update[pos_dst] = t

    def reset_state(self):
        zeros(self.last_update)

    def detach(self):
        return

    def forward(self, n_id):
        return self.last_update[n_id]
    

