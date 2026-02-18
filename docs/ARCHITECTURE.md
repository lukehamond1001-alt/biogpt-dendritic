# Architecture Deep Dive

This document explains the BioGPT architecture in detail, covering the biological motivation, implementation decisions, and how each component maps to neuroscience.

## Overview

BioGPT replaces the standard transformer attention block with a hierarchy of biologically-motivated components:

```
Standard Transformer:         BioGPT:
  LayerNorm                     LayerNorm
  Multi-Head Attention          CorticalLayer
  Residual                        -> Neuron 1 (EfficientBioNeuron)
  LayerNorm                          -> Branch 1..N (Multi-Head Attn)
  FFN                                -> Causal Temporal Conv1d
  Residual                           -> Per-Branch FFN (Grouped Conv1d)
                                     -> Per-Branch Norm (GroupNorm)
                                     -> Soma Projection
                                     -> Per-Neuron FFN
                                -> Neuron 2 ...
                                -> Weighted Combination (softmax)
                                Residual
```

## Component Details

### 1. EfficientBioNeuron

The core computational unit. Each neuron contains multiple **dendritic branches** that process input in parallel, then integrates them through a **soma**.

**Biological analog**: A pyramidal neuron in the neocortex. The dendrites (branches) receive input from different synapses, perform local computation, and send signals to the cell body (soma) for integration.

**Implementation**: All branches within a neuron are computed as a single batched operation using multi-head attention, where each attention head represents one dendritic branch.

```python
# Each branch = one attention head
# n_branches = number of heads
# d_branch = d_model // n_branches
qkv = self.qkv(x)  # Single projection for all branches
Q, K, V = split(qkv)
attn_out = scaled_dot_product_attention(Q, K, V, is_causal=True)
```

### 2. Causal Temporal Conv1d

**Biological analog**: Local dendritic integration -- nearby synaptic inputs on the same branch interact with each other over a short spatial/temporal window.

**Implementation**: A depthwise causal convolution with kernel_size=5, meaning each branch integrates information from the current and 4 previous positions:

```python
# groups=n_branches means each branch has its own filter
self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=5,
                                padding=4, groups=n_branches)
# Causal: trim to only use past + present
output = self.temporal_conv(input)[:, :, :T]
```

**Why not GRU?** We originally used GRU for temporal integration, but it's inherently sequential (each timestep depends on the previous). The Conv1d can process all timesteps in parallel, giving ~200x speedup on GPU.

### 3. Per-Branch FFN (Grouped Conv1d)

**Biological analog**: Dendrites perform nonlinear computation locally, not just passive signal transmission. Each branch (dendrite) has its own computational capacity.

**Implementation**: Using grouped Conv1d where each group corresponds to one branch:

```python
# groups=n_branches: each branch has independent weights
self.ff_up = nn.Conv1d(d_model, d_model * 2, kernel_size=1, groups=n_branches)
self.ff_down = nn.Conv1d(d_model * 2, d_model, kernel_size=1, groups=n_branches)
```

This gives each branch its own 2-layer FFN with GELU activation, but computed as a single CUDA kernel call.

### 4. Per-Branch GroupNorm

**Biological analog**: Local homeostatic regulation of dendritic activity. Each branch maintains its own activation statistics independently.

**Implementation**: GroupNorm with groups=n_branches. Critical detail: we normalize **per-position** to prevent information leakage across the sequence dimension:

```python
# Reshape to (B*T, D, 1) so GroupNorm only sees one position at a time
residual_bt = residual.reshape(B * T, d_model).unsqueeze(-1)
normed = self.branch_norm(residual_bt).squeeze(-1).reshape(B, T, d_model)
```

**Bug history**: Our initial implementation normalized across the full sequence `(B, D, T)`, which caused future-token information leakage. This was one of two critical causality bugs we discovered and fixed (the other was in the attention output reshape).

### 5. Soma Integration + Per-Neuron FFN

**Biological analog**: The cell body (soma) integrates signals from all dendrites and performs its own nonlinear processing before generating an output (action potential).

**Implementation**: A linear projection followed by a per-neuron feed-forward network:

```python
soma_out = self.soma_proj(normed)  # Integrate branches
soma_out = soma_out + self.soma_ffn(self.ln_soma(soma_out))  # Per-neuron computation
```

Each neuron has its own soma FFN -- this is **not** shared across neurons in a layer. This mimics the fact that each biological neuron has its own cell body with unique properties.

### 6. CorticalLayer (Multiple Neurons)

**Biological analog**: Cortical minicolumns -- groups of neurons that process the same input but specialize in different aspects.

**Implementation**: Each layer contains 2-3 neurons that process the same input in parallel. Their outputs are combined via learned softmax weights:

```python
weights = F.softmax(self.neuron_weights, dim=0)  # Competition
output = sum(w * proj(neuron_out) for w, neuron_out, proj in ...)
```

The softmax competition means neurons must learn complementary roles -- if one neuron dominates, others must contribute something different to earn weight.

### 7. Cortical Layout

**Biological analog**: The neocortex has a hierarchical organization from primary sensory areas (many small, fine-grained neurons) to higher association areas (fewer, larger neurons with broader receptive fields).

**Implementation**: We use a formula-based layout that varies neuron count and branch granularity by layer depth:

| Layer Zone | Layers | Neurons | Branches/Neuron | Branch Size | Biological Analog |
|---|---|---|---|---|---|
| Early | 0-3 | 2 | 8 | 96-dim | Primary sensory cortex |
| Middle | 4-7 | 3 | 8, 6, 4 | 96/128/192-dim | Association cortex |
| Late | 8-11 | 2 | 4 | 192-dim | Prefrontal / motor cortex |

Early layers have many fine-grained branches for detailed pattern detection. Late layers have fewer, larger branches for abstract reasoning.

## Efficiency Techniques

### Batched Operations

Every operation that was originally a Python loop over branches has been replaced with a single batched GPU operation:

| Original | Batched Replacement | Speedup |
|---|---|---|
| Loop over branches + per-branch attention | Multi-head attention | ~N_branches x |
| Loop over branches + per-branch FFN | Grouped Conv1d | ~N_branches x |
| Loop over branches + per-branch norm | GroupNorm | ~N_branches x |
| Sequential GRU per branch | Causal depthwise Conv1d | ~T x |

Combined, this achieved ~200x training speedup (from ~115 tok/s to ~24,500 tok/s).

### Gradient Checkpointing

Each CorticalLayer is wrapped in `torch.utils.checkpoint`, which recomputes activations during the backward pass instead of storing them. This trades ~30% more compute for ~60% less GPU memory, enabling the 294M parameter model to train on a single 24GB GPU.

### Mixed Precision

All computation uses bfloat16 mixed precision via `torch.amp.autocast`, reducing memory usage and enabling hardware-accelerated matrix operations on modern GPUs.

## Parameter Breakdown

For the default configuration (d_model=768, 12 layers):

| Component | Params | Notes |
|---|---|---|
| Token embedding | 38.6M | 50,304 x 768 |
| Position embedding | 0.8M | 1,024 x 768 |
| LM head | 38.6M | 768 x 50,304 |
| Early layers (0-3) | ~80M | 2 neurons x 8 branches each |
| Middle layers (4-7) | ~96M | 3 neurons x [8,6,4] branches |
| Late layers (8-11) | ~36M | 2 neurons x 4 branches each |
| **Total** | **~294M** | |
