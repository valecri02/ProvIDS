import copy
from typing import Callable, Dict, Tuple
import time
import torch
from torch import Tensor
from torch.nn import GRU, RNN, Linear
from typing import Callable, Optional
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from utils import get_indices
import wandb

class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        from torch_scatter import scatter_max
        # t contains the timestamp of the events
        # index contains an integer that represent the node to which the message belong
        # dim_size should be equal to max(index)+1 -> corresponds to the number of nodes
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) 
        # argmax contain a list with the position in "t" with the max value for each index
         
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out

class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean')
    

class SumAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')
    
class RNNAggregator(torch.nn.Module):
    def __init__(self, input_dim, output_dim, log):
        super().__init__()
        hidden_dim = 128
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = GRU(input_dim, hidden_dim, 1)
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        self.log = log

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        """
        msg: Messages (index_size * dimension_size) (200*10*200*2 * 327)
        index: Nodes to which the messages belong (previous_batch_size*neigbourhood_size * batch_size * 2) (200*10*200*2)
        t: timestamp of messages
        dim_size: Number of all nodes (nodes w/o messages)
        """
        out = msg.new_zeros((dim_size, msg.size(-1)))
        # get the grouping indices
        t0 = time.time()

        # sorting messages and indexes (node) by timestamp of the messages
        sort_indices = torch.sort(t)[1]
        index = index[sort_indices]
        msg = msg[sort_indices]
        t1 = time.time()
        # grouping messages belonging to the same node
        indices, msgs = get_indices(index, msg) 
        if self.log:
            wandb.log({"get_indices time": time.time() - t1})
        # create the node-wise message sequences
        # 
        if self.log:
            wandb.log({"sorting time": time.time() - t0})
        t0 = time.time()
        if len(indices) > 0:
            msgs = pad_sequence(msgs)
            if self.log:
                wandb.log({'msgs_size':msgs.size(0)*msgs.size(1)})

            # initial state: zeros
            hx = msg.new_zeros((1, len(indices), self.hidden_dim))
            # getting the updated hidden state (after getting msgs)
            # aggregating messages for the same node
            _, hn = self.rnn(msgs, hx)  # hn has the same size of hx
            # linear transforation to get the output (aggregation of the messages)
            hn = self.mlp(hn.squeeze())
            out[list(indices.keys())] = hn
        if self.log:
            wandb.log({"rnn time": time.time() - t0})

        return out
    
class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, edge_encoder: Optional[Callable] = None):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
        self.edge_encoder = edge_encoder

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor,
                t_enc: Tensor):
        if self.edge_encoder is not None:
            raw_msg = self.edge_encoder(raw_msg)
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)

class RawOnlyMessage(torch.nn.Module):
    """Message module that ignores memory embeddings and relies only on `raw_msg`.

    This is useful when `raw_msg` already contains the desired information, e.g.
    concatenated GNN embeddings: `[z_src_gnn | z_dst_gnn | edge_msg]`.
    """

    def __init__(self, raw_msg_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return torch.cat([raw_msg, t_enc], dim=-1)
    

AGGREGATOR_CONFS = {
    'rnn': RNNAggregator,
    'mean': MeanAggregator,
    'last': LastAggregator,
    'sum': SumAggregator,
}