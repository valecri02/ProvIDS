from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .layer import mLSTMLayer, mLSTMLayerConfig

MLSTMStateDictType = Dict[str, Any]


class mLSTMMemoryAdapter(torch.nn.Module):
    """Adapter wrapping mLSTMLayer.step for TGN node-memory updates.

    TGN produces one message vector per node after message aggregation. This
    adapter normalizes and projects that vector into the mLSTM embedding space,
    executes one recurrent mLSTM step, and returns both the exposed memory
    vector and the full internal recurrent state.
    """
    def __init__(self, message_dim: int, memory_dim: int, num_heads: int = 4,
                 context_length: int = 64, conv1d_kernel_size: int = 4):
        super().__init__()

        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.input_norm = torch.nn.LayerNorm(message_dim)
        self.input_proj = torch.nn.Linear(message_dim, memory_dim)

        config = mLSTMLayerConfig(
            embedding_dim=memory_dim,
            num_heads=num_heads,
            context_length=context_length,
            conv1d_kernel_size=conv1d_kernel_size,
            bias=False,
            dropout=0.0,
        )
        self.mlstm_layer = mLSTMLayer(config)

        inner_dim = self.mlstm_layer.config._inner_embedding_dim
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"mLSTM inner embedding dim={inner_dim} must be divisible "
                f"by mlstm_num_heads={num_heads}.")

    def forward(self, aggregated_msg: Tensor, prev_memory: Tensor,
                state: Optional[MLSTMStateDictType] = None
                ) -> Tuple[Tensor, MLSTMStateDictType]:
        """Run one mLSTM memory update step.

        Args:
            aggregated_msg: [batch_size, message_dim] aggregated messages.
            prev_memory: [batch_size, memory_dim] previous exposed memory.
                Currently accepted for GRUCell API compatibility.
            state: Optional full mLSTM layer state with `mlstm_state` and
                `conv_state`.

        Returns:
            new_memory: [batch_size, memory_dim] updated exposed memory.
            state: Updated full mLSTM layer recurrent state.
        """
        del prev_memory

        x = self.input_proj(self.input_norm(aggregated_msg)).unsqueeze(1)
        mlstm_state = None if state is None else state.get('mlstm_state')
        conv_state = None if state is None else state.get('conv_state')

        output, state_dict = self.mlstm_layer.step(
            x,
            mlstm_state=mlstm_state,
            conv_state=conv_state,
        )

        return output.squeeze(1), state_dict

    def reset_parameters(self):
        self.input_norm.reset_parameters()
        self.input_proj.reset_parameters()
        self.mlstm_layer.reset_parameters()
