import torch
import os
from typing import Callable, Optional, Dict, Tuple, Set
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import TGNMemory 
from torch import Tensor
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.utils import scatter
from .mlstm import MLSTMStateDictType, mLSTMMemoryAdapter

TGNMessageStoreType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
TGNMessageStoreWithZType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').lower() in {'1', 'true', 'yes'}


class IdentityLayer(torch.nn.Module):
    # Used by memory variants that do not have an RNN-based update.
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
                 message_batch: int = 200,
                 use_mlstm: bool = False,
                 mlstm_num_heads: int = 4,
                 mlstm_state_max_nodes: Optional[int] = None,
                 mlstm_debug: bool = False):
        self.message_batch = message_batch
        self.use_mlstm = use_mlstm
        self.mlstm_num_heads = mlstm_num_heads
        env_max_nodes = os.environ.get('PROVIDS_MLSTM_STATE_MAX_NODES')
        if mlstm_state_max_nodes is None and env_max_nodes:
            mlstm_state_max_nodes = int(env_max_nodes)
        self.mlstm_state_max_nodes = mlstm_state_max_nodes
        self.mlstm_debug = mlstm_debug or _env_flag('PROVIDS_MLSTM_DEBUG')
        self.mlstm_state_storage_dtype = self._resolve_mlstm_state_storage_dtype()

        super().__init__(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)
        
        # Initialize memory update module (GRU or mLSTM)
        if self.use_mlstm:
            self.gru = mLSTMMemoryAdapter(
                message_dim=message_module.out_channels,
                memory_dim=memory_dim,
                num_heads=self.mlstm_num_heads,
                context_length=64,
                conv1d_kernel_size=4
            )
        else:
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

    def _mlstm_log(self, msg: str):
        if self.use_mlstm and self.mlstm_debug:
            print(f"[mLSTM] {msg}")

    def _resolve_mlstm_state_storage_dtype(self) -> Optional[torch.dtype]:
        dtype_name = os.environ.get('PROVIDS_MLSTM_STATE_DTYPE', '').lower()
        if dtype_name in {'', 'float32', 'fp32'}:
            return None
        if dtype_name in {'float16', 'fp16', 'half'}:
            return torch.float16
        if dtype_name in {'bfloat16', 'bf16'}:
            return torch.bfloat16
        raise ValueError(
            "PROVIDS_MLSTM_STATE_DTYPE must be one of: float32, float16, bfloat16")

    def _store_mlstm_tensor_cpu(self, tensor: Tensor) -> Tensor:
        tensor = tensor.detach().cpu().clone()
        if self.mlstm_state_storage_dtype is not None:
            tensor = tensor.to(dtype=self.mlstm_state_storage_dtype)
        return tensor

    def _reset_message_store_z(self):
        device = self.memory.device
        i = self.memory.new_empty((0, ), device=device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=device)
        z = self.memory.new_empty((0, self.memory.size(-1)), device=device)
        # Message store format: (src, dst, t, msg, z_src, z_dst)
        self.msg_s_store_z: TGNMessageStoreWithZType = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}
        self.msg_d_store_z: TGNMessageStoreWithZType = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}

    def _init_mlstm_states(self):
        """Initialize sparse dictionary storage for per-node mLSTM states."""
        if not hasattr(self, '_mlstm_state_dict'):
            self._mlstm_state_dict: Dict[int, MLSTMStateDictType] = {}
        if not hasattr(self, '_pending_mlstm_node_ids'):
            self._pending_mlstm_node_ids: Set[int] = set()

    def _empty_mlstm_state(self, batch_size: int) -> MLSTMStateDictType:
        """Create a dense zero state batch on the active memory device."""
        mlstm_config = self.gru.mlstm_layer.config
        NH = mlstm_config.num_heads
        inner_dim = mlstm_config._inner_embedding_dim
        if inner_dim % NH != 0:
            raise ValueError(
                f"mLSTM inner embedding dim={inner_dim} must be divisible "
                f"by mlstm_num_heads={NH}.")

        DH = inner_dim // NH
        device = self.memory.device
        dtype = self.memory.dtype

        mlstm_state = (
            torch.zeros(batch_size, NH, DH, DH, device=device, dtype=dtype),
            torch.zeros(batch_size, NH, DH, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, NH, 1, 1, device=device, dtype=dtype),
        )

        conv_state = None
        kernel_size = self.gru.mlstm_layer.conv1d.config.kernel_size
        if kernel_size > 0:
            conv_state = (
                torch.zeros(batch_size, kernel_size, inner_dim,
                            device=device, dtype=dtype),
            )

        return {'mlstm_state': mlstm_state, 'conv_state': conv_state}
    
    def _get_mlstm_state(self, n_id: Tensor) -> MLSTMStateDictType:
        """Get full mLSTM layer states, creating zeros for unseen nodes."""
        self._init_mlstm_states()
        device = self.memory.device
        dtype = self.memory.dtype

        state = self._empty_mlstm_state(n_id.size(0))
        c_state, n_state, m_state = state['mlstm_state']
        conv_state = state['conv_state']

        for i, node in enumerate(n_id.tolist()):
            stored = self._mlstm_state_dict.get(node)
            if stored is None:
                continue

            c_i, n_i, m_i = stored['mlstm_state']
            c_state[i] = c_i.to(device=device, dtype=dtype)
            n_state[i] = n_i.to(device=device, dtype=dtype)
            m_state[i] = m_i.to(device=device, dtype=dtype)

            stored_conv_state = stored.get('conv_state')
            if conv_state is not None and stored_conv_state is not None:
                conv_state[0][i] = stored_conv_state[0].to(device=device, dtype=dtype)

        self._mlstm_log(
            f"loaded state for {sum(1 for node in n_id.tolist() if node in self._mlstm_state_dict)}/"
            f"{n_id.numel()} nodes; cached={len(self._mlstm_state_dict)}")
        return state
    
    def _set_mlstm_state(self, n_id: Tensor, state: MLSTMStateDictType):
        """Store updated full mLSTM layer states in a sparse CPU dictionary."""
        self._init_mlstm_states()

        new_nodes = set(n_id.tolist()).difference(self._mlstm_state_dict.keys())
        if (self.mlstm_state_max_nodes is not None
                and len(self._mlstm_state_dict) + len(new_nodes) > self.mlstm_state_max_nodes):
            raise RuntimeError(
                f"mLSTM state cache would exceed mlstm_state_max_nodes="
                f"{self.mlstm_state_max_nodes}.")

        c_state, n_state, m_state = state['mlstm_state']
        conv_state = state.get('conv_state')
        for i, node in enumerate(n_id.tolist()):
            # Keep sparse recurrent cache on CPU to avoid long-lived GPU memory
            # growth; move back to device on demand in _get_mlstm_state.
            node_conv_state = None
            if conv_state is not None:
                node_conv_state = (self._store_mlstm_tensor_cpu(conv_state[0][i]),)

            self._mlstm_state_dict[node] = (
                {
                    'mlstm_state': (
                        self._store_mlstm_tensor_cpu(c_state[i]),
                        self._store_mlstm_tensor_cpu(n_state[i]),
                        self._store_mlstm_tensor_cpu(m_state[i]),
                    ),
                    'conv_state': node_conv_state,
                }
            )

        self._mlstm_log(
            f"stored state for {n_id.numel()} nodes; cached={len(self._mlstm_state_dict)};")

    def _add_pending_mlstm_nodes(self, n_id: Tensor):
        if self.use_mlstm:
            self._init_mlstm_states()
            self._pending_mlstm_node_ids.update(int(i) for i in n_id.tolist())

    def _discard_pending_mlstm_nodes(self, n_id: Tensor):
        if self.use_mlstm:
            self._init_mlstm_states()
            self._pending_mlstm_node_ids.difference_update(int(i) for i in n_id.tolist())

    def _pending_mlstm_nodes_tensor(self) -> Tensor:
        self._init_mlstm_states()
        if len(self._pending_mlstm_node_ids) == 0:
            return torch.empty(0, dtype=torch.long, device=self.memory.device)
        return torch.tensor(sorted(self._pending_mlstm_node_ids),
                            dtype=torch.long, device=self.memory.device)
    
    def reset_mlstm_states(self):
        """Reset all mLSTM states by clearing the state dictionary."""
        if self.use_mlstm:
            self._mlstm_state_dict = {}
            self._pending_mlstm_node_ids = set()
            self._mlstm_log("reset cached states")

    def reset_state(self):
        super().reset_state()
        self._reset_message_store_z()
        if self.use_mlstm:
            self.reset_mlstm_states()
        if self.use_mlstm:
            self.reset_mlstm_states()

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

        device = self.memory.device
        src = torch.cat(src, dim=0).to(device)
        dst = torch.cat(dst, dim=0).to(device)
        t = torch.cat(t, dim=0).to(device)

        # Filter out empty tensors to avoid `invalid configuration argument`.
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        z_src = [m for i, m in enumerate(z_src) if m.numel() > 0 or i == 0]
        z_dst = [m for i, m in enumerate(z_dst) if m.numel() > 0 or i == 0]

        raw_msg = torch.cat(raw_msg, dim=0).to(device)
        z_src = torch.cat(z_src, dim=0).to(device)
        z_dst = torch.cat(z_dst, dim=0).to(device)

        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(z_src, z_dst, raw_msg, t_enc)
        return msg, t, src, dst

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(
            n_id, commit_mlstm_state=self.use_mlstm)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor,
                            commit_mlstm_state: bool = False):
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        
        if getattr(self, '_store_mode', 'base') == 'z':
            # z mode: use stored GNN embeddings
            msg_s, t_s, src_s, dst_s = self._compute_msg_z(n_id, self.msg_s_store_z, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg_z(n_id, self.msg_d_store_z, self.msg_d_module)
        else:
            # base mode: use internal message stores
            msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)
        
        
        if getattr(self, '_store_mode', 'base') == 'z':
            # z mode: use stored GNN embeddings
            msg_s, t_s, src_s, dst_s = self._compute_msg_z(n_id, self.msg_s_store_z, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg_z(n_id, self.msg_d_store_z, self.msg_d_module)
        else:
            # base mode: use internal message stores
            msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)
        
        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Update memory: mLSTM or GRU
        if self.use_mlstm:
            state = self._get_mlstm_state(n_id)
            memory, state_new = self.gru(aggr, self.memory[n_id], state)
            if commit_mlstm_state:
                self._set_mlstm_state(n_id, state_new)
        else:
            memory = self.gru(aggr, self.memory[n_id])

        # Get last updates.
        # Get last updates.
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
            self._discard_pending_mlstm_nodes(n_id)
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            # Reverse direction store: (dst -> src) with swapped embeddings.
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
            self._add_pending_mlstm_nodes(n_id)
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
            if self.use_mlstm:
                n_id = self._pending_mlstm_nodes_tensor()
                flush_batch = min(self.message_batch, 32)
                self._mlstm_log(
                    f"train->eval flush pending={n_id.numel()} batch_size={flush_batch}")
                for i in range(0, n_id.size(0), flush_batch):
                    self._update_memory(n_id[i:i + flush_batch])
                self._pending_mlstm_node_ids.clear()
                self._mlstm_log(f"train->eval flush complete; cached={len(self._mlstm_state_dict)};")
            else:
                # Preserve PyG's GRU behavior: all nodes are flushed.
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
        n_id = self._last_updated_n_id
        if self.training and self.use_mlstm:
            self._update_memory(n_id)
            self._discard_pending_mlstm_nodes(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._add_pending_mlstm_nodes(n_id)
            return
        return super().update_state(src, dst, t, raw_msg)

    def detach(self):
        """Detaches the memory *and* stored messages from gradient computation.

        PyG's :class:`~torch_geometric.nn.TGNMemory` only detaches `self.memory`.
        However, the per-node message stores (`msg_s_store`, `msg_d_store`) may
        contain tensors that keep the autograd graph alive across batches.
        For mLSTM, we also detach states to free GPU memory.
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
    
