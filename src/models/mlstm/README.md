# mLSTM for ProvIDS

This is a trimmed copy of the mLSTM module from [NX-AI/xlstm](https://github.com/NX-AI/xlstm), optimized for integration into ProvIDS memory layers.

## Architecture

- **`mLSTMCell`** — core multiplicative LSTM computation with both `forward` (full sequence, parallel) and `step` (single-step, recurrent) modes
- **`mLSTMLayer`** — full mLSTM block including projection layers, causal convolution, normalization, and gating
- **`backends.py`** — parallel and recurrent kernel implementations

## Current Integration Plan

**Status**: Step-based integration (recurrent mode for streaming node updates)

For ProvIDS memory, we use:
- `mLSTMLayer.step(x, mlstm_state, conv_state)` to process one message per node incrementally
- Per-node persistent storage of `mlstm_state` and `conv_state` across batch updates
- Forward-pass support can be added later for full-sequence training or validation

## Dependencies

Requires:
- PyTorch