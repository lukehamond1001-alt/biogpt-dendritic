"""
BioGPT -- Biologically-Inspired Dendritic Language Model

A novel transformer variant where standard attention layers are replaced by
biologically-motivated dendritic neurons organized into cortical columns.

Architecture Hierarchy:
    Branch (dendrite) -> Neuron (soma) -> Layer (cortical column)

Features per Branch (batched as multi-head attention + grouped ops):
    - Q/K/V causal attention (batched as multi-head attention)
    - Per-branch FFN via grouped Conv1d (each branch has own weights)
    - Per-branch LayerNorm via GroupNorm
    - Causal temporal Conv1d (local dendritic integration, parallel)

Features per Neuron:
    - Multiple branches with uniform size (varies across neurons/layers)
    - Soma projection (combines branch outputs)
    - Per-neuron soma FFN (each neuron has its own cell body)

Features per Layer:
    - Multiple neurons (cortical column)
    - Weighted combination of neuron outputs (softmax competition)
    - Cortical layout: early=many small branches, late=few large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import math
from typing import Optional, List, Tuple


class EfficientBioNeuron(nn.Module):
    """
    One biological neuron with all dendritic branches computed in parallel.

    Uses batched GPU operations instead of Python loops:
      - Multi-head attention: single QKV projection -> parallel attention heads
      - Grouped Conv1d: per-branch FFN with separate weights, one CUDA kernel
      - GroupNorm: per-branch normalization in one call
      - Causal temporal Conv1d: parallel local integration per branch

    Each neuron has uniform branch size (d_branch = d_model // n_branches)
    but different neurons can have different numbers/sizes of branches.
    """

    def __init__(self, d_model: int, n_branches: int,
                 soma_ffn_expansion: float = 3.0, dropout: float = 0.0,
                 nmda_tau: float = 0.9):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches
        self.d_branch = d_model // n_branches
        assert d_model % n_branches == 0, \
            f"d_model ({d_model}) must be divisible by n_branches ({n_branches})"

        d_b = self.d_branch

        # Batched attention (all branches at once)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_dropout = dropout

        # NMDA temporal trace (per-branch, batched)
        self.nmda_decay = nn.Parameter(
            torch.randn(1, n_branches, 1, d_b) * 0.1 + nmda_tau
        )

        # Causal temporal Conv1d (parallel alternative to GRU)
        # Local dendritic integration over nearby timesteps
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=4,
            groups=n_branches,
        )

        # Per-branch FFN via grouped Conv1d
        self.ff_up = nn.Conv1d(d_model, d_model * 2, kernel_size=1,
                               groups=n_branches)
        self.ff_down = nn.Conv1d(d_model * 2, d_model, kernel_size=1,
                                 groups=n_branches)
        self.ff_dropout = nn.Dropout(dropout)

        # Per-branch norm via GroupNorm
        self.branch_norm = nn.GroupNorm(n_branches, d_model)

        # Soma (integrates branches -> d_model output)
        self.soma_proj = nn.Linear(d_model, d_model, bias=False)
        self.soma_dropout = nn.Dropout(dropout)

        # Per-neuron soma FFN
        self.ln_soma = nn.LayerNorm(d_model)
        soma_hidden = int(d_model * soma_ffn_expansion)
        self.soma_ffn = nn.Sequential(
            nn.Linear(d_model, soma_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(soma_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, states=None):
        B, T, D = x.shape
        n_h = self.n_branches
        d_h = self.d_branch

        # 1. Multi-head attention (all branches at once)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, n_h, d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2. NMDA trace (enriches V with temporal memory)
        if states is not None and states.get('nmda') is not None:
            nmda_trace = states['nmda']
            tau = torch.sigmoid(self.nmda_decay)
            V = V + nmda_trace.unsqueeze(2) * tau

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # 3. Causal temporal Conv1d (parallel temporal integration)
        # Transpose heads and time before flattening (critical for causality)
        branch_out = attn_out.transpose(1, 2).contiguous()
        branch_flat = branch_out.reshape(B, T, n_h * d_h)
        branch_cf = branch_flat.permute(0, 2, 1)

        # Causal conv: apply then trim future tokens
        temporal_out = self.temporal_conv(branch_cf)[:, :, :T]
        branch_cf = branch_cf + temporal_out

        # 4. Per-branch FFN
        ff_hidden = F.gelu(self.ff_up(branch_cf))
        ff_out = self.ff_down(self.ff_dropout(ff_hidden))
        ff_out = ff_out.permute(0, 2, 1)

        # 5. Per-branch norm via GroupNorm (per-position, no cross-position leakage)
        residual = branch_flat + ff_out
        residual_bt = residual.reshape(B * T, self.d_model).unsqueeze(-1)
        normed = self.branch_norm(residual_bt).squeeze(-1).reshape(B, T, self.d_model)

        # 6. Update NMDA trace
        normed_heads = normed.reshape(B, T, n_h, d_h)
        branch_last = normed_heads[:, -1, :, :]
        tau = torch.sigmoid(self.nmda_decay.squeeze(2))

        if states is not None and states.get('nmda') is not None:
            old_nmda = states['nmda']
        else:
            old_nmda = torch.zeros(B, n_h, d_h, device=x.device, dtype=x.dtype)
        new_nmda = tau * old_nmda + (1 - tau) * branch_last

        # 7. Soma integration
        soma_out = self.soma_dropout(self.soma_proj(normed))
        soma_out = soma_out + self.soma_ffn(self.ln_soma(soma_out))

        new_states = {'nmda': new_nmda}
        return soma_out, new_states


class CorticalLayer(nn.Module):
    """
    One cortical layer -- multiple neurons firing in parallel.

    Neurons produce independent outputs, combined via learned softmax weights.
    Pre-norm architecture with residual connection.
    """

    def __init__(self, d_model: int, neuron_configs: List[int],
                 soma_ffn_expansion: float = 3.0, dropout: float = 0.0,
                 nmda_tau: float = 0.9):
        super().__init__()
        self.n_neurons = len(neuron_configs)

        self.ln = nn.LayerNorm(d_model)

        self.neurons = nn.ModuleList([
            EfficientBioNeuron(d_model, n_branches, soma_ffn_expansion,
                               dropout, nmda_tau)
            for n_branches in neuron_configs
        ])

        self.neuron_weights = nn.Parameter(torch.zeros(self.n_neurons))

        self.neuron_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(self.n_neurons)
        ])

    def forward(self, x, states=None):
        if states is None:
            states = [None] * self.n_neurons

        h = self.ln(x)
        weights = F.softmax(self.neuron_weights, dim=0)

        combined = torch.zeros_like(x)
        new_states = []

        for i, neuron in enumerate(self.neurons):
            n_out, n_state = neuron(h, states[i])
            combined = combined + weights[i] * self.neuron_projs[i](n_out)
            new_states.append(n_state)

        return x + combined, new_states


class GatedFeedback(nn.Module):
    """
    Feedback from a deeper layer to a shallower layer.

    Inspired by cortical Layer 5/6 -> Layer 2/3 top-down connections.
    A learned sigmoid gate controls how much feedback to incorporate.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, target, feedback):
        fb = self.proj(feedback)
        gate_input = torch.cat([target, fb], dim=-1)
        gate = self.gate(gate_input)
        return target + gate * self.ln(fb)


def generate_cortical_layout(d_model: int = 768, n_layers: int = 12):
    """
    Generate per-layer neuron and branch configurations
    based on cortical biological ratios.

    Early layers: 2 neurons, many smaller branches (sensory/pattern detection)
    Middle layers: 3 neurons, mixed branch counts (association/integration)
    Late layers: 2 neurons, few large branches (abstract reasoning/output)
    """
    layouts = []

    for layer_idx in range(n_layers):
        progress = layer_idx / max(n_layers - 1, 1)

        if progress < 0.33:
            n_branches_per_neuron = 8
            while d_model % n_branches_per_neuron != 0:
                n_branches_per_neuron -= 1
            neuron_configs = [n_branches_per_neuron, n_branches_per_neuron]

        elif progress > 0.67:
            n_branches_per_neuron = 4
            while d_model % n_branches_per_neuron != 0:
                n_branches_per_neuron -= 1
            neuron_configs = [n_branches_per_neuron, n_branches_per_neuron]

        else:
            n_many, n_med, n_few = 8, 6, 4
            while d_model % n_many != 0:
                n_many -= 1
            while d_model % n_med != 0:
                n_med -= 1
            while d_model % n_few != 0:
                n_few -= 1
            neuron_configs = [n_many, n_med, n_few]

        layouts.append(neuron_configs)

    return layouts


def generate_feedback_pairs(n_layers: int = 12):
    """
    Generate feedback connection pairs (source -> target).

    Late layers (last 1/3) send feedback to early layers (first 1/3),
    mimicking cortical Layer 5/6 -> Layer 2/3 connections.
    """
    early_end = n_layers // 3
    late_start = n_layers - n_layers // 3

    pairs = []
    for source in range(late_start, n_layers):
        target = source - late_start
        if target < early_end:
            pairs.append((source, target))

    return pairs


class BioGPT(nn.Module):
    """
    Full biological dendritic GPT model.

    Combines:
      - Dendritic branches with NMDA + temporal conv + FFN + norm (batched)
      - Multiple neurons per layer with per-neuron soma FFN
      - Cortical layout (structured variation across layers)
      - Gated feedback from deep to early layers
      - Optional multi-pass refinement
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 12,
        soma_ffn_expansion: float = 3.0,
        dropout: float = 0.0,
        nmda_tau: float = 0.9,
        max_seq_len: int = 1024,
        n_passes: int = 1,
        cortical_layout: List = None,
        feedback_pairs: List = None,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_passes = n_passes
        self.vocab_size = vocab_size
        self.use_checkpoint = use_checkpoint

        self.config = {
            "vocab_size": vocab_size, "d_model": d_model,
            "n_layers": n_layers, "soma_ffn_expansion": soma_ffn_expansion,
            "dropout": dropout, "nmda_tau": nmda_tau,
            "max_seq_len": max_seq_len, "n_passes": n_passes,
            "cortical_layout": cortical_layout,
            "feedback_pairs": feedback_pairs,
            "use_checkpoint": use_checkpoint,
        }

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        if cortical_layout is None:
            cortical_layout = generate_cortical_layout(d_model, n_layers)
        self.cortical_layout = cortical_layout

        self.layers = nn.ModuleList([
            CorticalLayer(d_model, cortical_layout[i],
                          soma_ffn_expansion, dropout, nmda_tau)
            for i in range(n_layers)
        ])

        if feedback_pairs is None:
            feedback_pairs = generate_feedback_pairs(n_layers)
        self.feedback_pairs = feedback_pairs

        self.feedback_connections = nn.ModuleDict()
        for source, target in feedback_pairs:
            key = f"fb_{source}_to_{target}"
            self.feedback_connections[key] = GatedFeedback(d_model)

        if n_passes > 1:
            self.pass_weights = nn.Parameter(torch.zeros(n_passes))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'nmda' not in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _run_layer_fn(self, layer, x):
        out, _ = layer(x, None)
        return out

    def _run_layers(self, x, use_states=False):
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                x = torch_checkpoint(
                    self._run_layer_fn, layer, x,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(x, None)
            layer_outputs.append(x)

        return x, layer_outputs

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Seq {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb_dropout(self.token_emb(idx) + self.pos_emb(pos))

        if self.n_passes == 1:
            x, _ = self._run_layers(x)
        else:
            pass_outputs = []

            x_pass1, layer_outs_1 = self._run_layers(x)
            pass_outputs.append(x_pass1)

            x_feedback = x.clone()
            for source, target in self.feedback_pairs:
                key = f"fb_{source}_to_{target}"
                if key in self.feedback_connections:
                    x_feedback = self.feedback_connections[key](
                        x_feedback, layer_outs_1[source])

            x_pass2, _ = self._run_layers(x_feedback)
            pass_outputs.append(x_pass2)

            weights = F.softmax(self.pass_weights, dim=0)
            x = sum(w * p for w, p in zip(weights, pass_outputs))

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_biogpt(**overrides) -> BioGPT:
    """Create BioGPT with default config matched for Pythia comparison."""
    kwargs = dict(
        vocab_size=50304,
        d_model=768,
        n_layers=12,
        soma_ffn_expansion=3.0,
        dropout=0.0,
        nmda_tau=0.9,
        max_seq_len=1024,
        n_passes=1,
        use_checkpoint=True,
    )
    kwargs.update(overrides)
    return BioGPT(**kwargs)


if __name__ == "__main__":
    print("=" * 70)
    print("  BioGPT -- Biologically-Inspired Dendritic Language Model")
    print("=" * 70)

    layout = generate_cortical_layout(768, 12)

    total_branches = 0
    print(f"\n  Cortical Layout:")
    for i, neuron_configs in enumerate(layout):
        progress = i / 11
        zone = "EARLY" if progress < 0.33 else (
            "LATE" if progress > 0.67 else "MID")
        n_neurons = len(neuron_configs)
        branches = sum(neuron_configs)
        total_branches += branches
        branch_info = [f"{nb}x{768//nb}" for nb in neuron_configs]
        print(f"    Layer {i:2d} [{zone:>5s}]: {n_neurons} neurons | "
              f"{branches} branches | {branch_info}")

    print(f"    Total branches: {total_branches}")

    print(f"\n  Building model...")
    model = create_biogpt()
    total = count_parameters(model)

    emb_params = sum(p.numel() for p in model.token_emb.parameters())
    emb_params += sum(p.numel() for p in model.pos_emb.parameters())
    emb_params += sum(p.numel() for p in model.lm_head.parameters())
    non_emb = total - emb_params

    print(f"\n  Pythia-160M target:  162,322,944 total")
    print(f"  BioGPT total:        {total:>12,}")
    print(f"  BioGPT embedding:    {emb_params:>12,}")
    print(f"  BioGPT non-emb:      {non_emb:>12,}")
    print(f"  vs Pythia:           {total - 162_322_944:>+12,} "
          f"({(total - 162_322_944)/162_322_944:+.1%})")

    print(f"\n  Done!")
