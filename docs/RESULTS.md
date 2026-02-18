# Detailed Results Analysis

This document provides a comprehensive analysis of all experimental results.

## Experimental Setup

### Models

| | BioGPT | Pythia-160M |
|---|---|---|
| Architecture | Dendritic (this work) | Standard transformer |
| Total parameters | 294M | 162M |
| Layers | 12 | 12 |
| d_model | 768 | 768 |
| Attention heads | Varies (4-8 per neuron) | 12 |
| Neurons per layer | 2-3 | N/A |
| Max sequence length | 1,024 | 2,048 |
| Tokenizer | GPT-NeoX-20B | GPT-NeoX-20B |
| Training data | The Pile | The Pile |
| Training tokens | 2B | 2B (step-1000 checkpoint) |

### Training Configuration

Both models were trained/evaluated with:
- **Optimizer**: AdamW with betas (0.9, 0.95) and weight decay 0.1
- **Learning rate**: Cosine schedule, 6e-4 max with 500-step warmup
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Precision**: bfloat16 mixed precision

## Phase 3: BioGPT vs Pythia-160M

### Baseline Performance

| Model | Parameters | Val Loss | Perplexity |
|---|---|---|---|
| BioGPT (unpruned) | 294M | 2.86 | 17.5 |
| Pythia-160M (unpruned) | 162M | 3.45 | 31.5 |

BioGPT achieves substantially lower perplexity, but it also has ~1.8x more parameters. The real test is whether this advantage holds after pruning to equal size.

### Pruning Curve: BioGPT

Magnitude-based pruning with 500 steps of masked fine-tuning at each level:

| Sparsity | Active Params | Val Loss | Perplexity | Loss Delta |
|---|---|---|---|---|
| 0% | 294M | 2.86 | 17.5 | baseline |
| 25% | 231M | 2.92 | 18.5 | +0.06 |
| 45% | 180M | 3.05 | 21.2 | +0.19 |
| 70% | 116M | 4.12 | 61.7 | +1.26 |
| 90% | 65M | 6.99 | 1,084 | +4.13 |

BioGPT shows remarkable stability through 25% pruning (+0.06 loss) and reasonable degradation at 45% (+0.19). The model only starts to seriously degrade at 70% sparsity.

### Pruning Curve: Pythia-160M

Same magnitude pruning methodology:

| Sparsity | Active Params | Val Loss | Perplexity | Loss Delta |
|---|---|---|---|---|
| 0% | 162M | 3.45 | 31.5 | baseline |
| 25% | 141M | 3.63 | 37.7 | +0.18 |
| 50% | 120M | 3.72 | 41.4 | +0.27 |
| 70% | 103M | 4.67 | 106.8 | +1.22 |
| 90% | 86M | 6.79 | 889 | +3.34 |

Pythia degrades faster: at 25% pruning it already loses +0.18 (3x worse than BioGPT's +0.06).

### Iso-Parameter Comparison

The headline result -- comparing at roughly equal parameter counts:

| Model | Active Params | Val Loss | Perplexity |
|---|---|---|---|
| **BioGPT (45% pruned)** | **180M** | **3.05** | **21.2** |
| **Pythia (unpruned)** | **162M** | **3.45** | **31.5** |

BioGPT wins by:
- **0.40 lower loss**
- **10.3 lower perplexity**
- Despite having ~45% of its weights removed

This suggests the dendritic architecture distributes knowledge more efficiently across its weights.

### Why Is BioGPT More Pruning-Resilient?

Several architectural features likely contribute:

1. **Redundant pathways**: Multiple neurons per layer mean pruning one neuron's branches doesn't eliminate the layer's capacity entirely
2. **Distributed representations**: Per-branch FFNs force knowledge to be distributed across branches rather than concentrated in a single large FFN
3. **Cortical hierarchy**: Different branch granularities at different depths create a natural diversity of representation scales
4. **Biological parallel**: This mirrors synaptic pruning in the developing brain -- the brain starts with far more synapses than it needs, and pruning strengthens the remaining connections

## Generation Quality Assessment

Beyond loss and perplexity, I personally read through every generation output from both models at each pruning level. Each model received 5 prompts (knowledge, reasoning, code, science, instruction) at every sparsity point, plus 12 prompts in the unpruned head-to-head.

### Methodology

Each model was given the same prompt and generated up to 40-120 tokens (depending on prompt type) with temperature=0.7 and top_k=40. I read every output and assessed coherence, relevance, structure, and whether the model maintained meaningful language vs. degenerating into repetition or gibberish.

### Findings by Pruning Level

**Unpruned (0%)**: Both models produce grammatically correct English. Neither gives precise factual answers (expected for non-instruction-tuned base models). BioGPT writes slightly more natural prose. Pythia shows more self-repetition. Code outputs have structure (return statements, indentation) but wrong logic for both.

**25% pruned**: BioGPT shows no visible degradation. Pythia's code outputs start breaking (`_c_c_c_c` repetition patterns). BioGPT clearly better.

**45-50% pruned (iso-parameter comparison)**: This is the decisive level. BioGPT (180M) still produces readable, on-topic text with proper code structure (`while` loops, `len()`). Science outputs discuss specific topics (biofuels, plant-derived medium). Pythia (120M) collapses into repetition loops (`with_k_k_k = 0;`, `Fordn:` repeated). BioGPT is clearly superior.

**70% pruned**: Both degrading. BioGPT maintains grammar but becomes repetitive ("effect of the effect of the effect"). Pythia's instruction output becomes `S.S.S.S.S.S.` symbol spam. BioGPT holds up slightly better.

**90% pruned**: Both models produce gibberish. Neither is functional.

### Summary

| Pruning Level | BioGPT Quality | Pythia Quality | Winner |
|---|---|---|---|
| 0% (unpruned) | Coherent, on-topic | Coherent, slightly repetitive | Tie |
| 25% | No visible degradation | Code breaking, repetition starting | BioGPT |
| 45-50% | Readable, structured code | Collapsed repetition loops | BioGPT |
| 70% | Repetitive but grammatical | Broken syntax, symbol spam | BioGPT |
| 90% | Gibberish | Gibberish | Tie (both dead) |

The qualitative assessment strongly reinforces the quantitative results: BioGPT's dendritic architecture degrades more gracefully under pruning, maintaining coherent language structure at sparsity levels where Pythia collapses into meaningless repetition.

## Limitations and Caveats

1. **Single training run**: Results from one training run each. Statistical significance would require multiple seeds.
2. **Parameter mismatch**: BioGPT starts at 294M and is pruned to 180M; Pythia starts at 162M. While we compare at similar active parameter counts, the architectures are fundamentally different in how they use those parameters.
3. **Limited evaluation**: We measure validation loss on The Pile only. Standard NLP benchmarks (HellaSwag, PIQA, ARC, etc.) would provide more comprehensive evaluation.
4. **Unstructured pruning**: Our magnitude pruning removes individual weights but doesn't reduce compute cost. Structured pruning (removing entire branches/neurons) would be needed for actual inference speedup.
