# Reproduction Guide

This guide covers everything needed to reproduce the BioGPT results from scratch.

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 16GB VRAM (A100/V100) | 24GB VRAM (RTX 4090/A100) |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 100GB free |
| Training time | ~30 hours | ~22 hours (RTX 4090) |

## Software Requirements

- Python 3.10+
- CUDA 12.0+
- PyTorch 2.0+

## Step-by-Step Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/lukehamond1001-alt/biogpt-dendritic.git
cd biogpt-dendritic

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Architecture

Run the model self-test to confirm parameter counts and architecture:

```bash
python -m biogpt.model
```

Expected output (key lines):
```
  Cortical Layout:
    Layer  0 [EARLY]: 2 neurons | 16 branches | ['8x96', '8x96']
    Layer  4 [  MID]: 3 neurons | 18 branches | ['8x96', '6x128', '4x192']
    Layer  8 [ LATE]: 2 neurons |  8 branches | ['4x192', '4x192']
    Total branches: 168

  Feedback connections:
    Layer 8 -> Layer 0
    Layer 9 -> Layer 1
    Layer 10 -> Layer 2
    Layer 11 -> Layer 3

  BioGPT total:      ~294,500,000
```

### 3. Prepare Training Data

Stream and tokenize 2B tokens from The Pile:

```bash
PYTHONUNBUFFERED=1 python -m biogpt.prepare_data \
    --target_tokens 2000000000 \
    --out_dir data
```

This will:
- Download from The Pile via HuggingFace streaming
- Tokenize with the GPT-NeoX-20B tokenizer (same as Pythia)
- Write to memory-mapped `train.bin` and `val.bin`
- Save progress periodically (resume-capable if interrupted)

**Expected time**: 2-4 hours (depends on network speed)
**Expected output**: `data/train.bin` (~3.7GB) and `data/val.bin` (~19MB)

### 4. Train BioGPT

```bash
PYTHONUNBUFFERED=1 python -m biogpt.train \
    --data_dir data \
    --out_dir checkpoints \
    --target_tokens 2000000000 \
    --block_size 1024 \
    --batch_size 4 \
    --grad_accum 8 \
    --n_passes 1
```

**Expected behavior**:
- ~24,500 tokens/second on RTX 4090
- Loss starts at ~10.5, drops to ~3.0 range by end
- Checkpoints saved every 500 eval steps
- Final checkpoint: `checkpoints/biogpt_final.pt`

**Expected time**: ~22 hours on RTX 4090

#### Optional: Timing Test First

To estimate training time on your hardware:

```bash
python -m biogpt.train --data_dir data --timing_test --timing_steps 50
```

### 5. Run Comparison Against Pythia

```bash
PYTHONUNBUFFERED=1 python -m biogpt.compare \
    --data_dir data \
    --biogpt_ckpt checkpoints/biogpt_final.pt \
    --output_dir results
```

This downloads Pythia-160M (step-1000 = 2B tokens), runs:
1. Head-to-head generation on 12 prompts
2. Validation loss comparison
3. BioGPT pruning curve: 0%, 25%, 45%, 70%, 90%
4. Pythia pruning curve: 0%, 25%, 50%, 70%, 90%

**Expected time**: ~2 hours
**Expected output**: `results/full_comparison.json`

### 6. Generate Charts

```bash
python scripts/generate_charts.py
```

### 7. Interactive Generation

```bash
python -m biogpt.generate \
    --checkpoint checkpoints/biogpt_final.pt \
    --interactive
```

## Expected Results

Your results should be approximately:

| Model | Active Params | Val Loss | PPL |
|---|---|---|---|
| BioGPT 0% | ~294M | ~2.86 | ~17.5 |
| BioGPT 45% | ~180M | ~3.05 | ~21.2 |
| Pythia 0% | ~162M | ~3.45 | ~31.5 |
| Pythia 50% | ~120M | ~3.72 | ~41.4 |

Exact numbers will vary due to random initialization, data sampling order, and hardware differences, but the relative ordering (BioGPT 45% beats Pythia 0%) should be consistent.

## Troubleshooting

### Out of Memory

If you get CUDA OOM errors:
- Reduce `--batch_size` to 2
- Increase `--grad_accum` to 16 (to maintain effective batch size)
- Ensure gradient checkpointing is enabled (default)

### Data Download Failures

The data preparation script has built-in retry logic. If it fails:
- Re-run the same command -- it will resume from where it left off
- If one data source fails, it automatically tries alternatives

### Slow Training

Expected throughput on different GPUs:
- RTX 4090: ~24,500 tok/s (~22 hours for 2B tokens)
- A100 40GB: ~20,000 tok/s (~27 hours)
- RTX 3090: ~15,000 tok/s (~37 hours)

If significantly slower, check that:
- Mixed precision is active (requires CUDA GPU)
- No other processes are using the GPU
- `PYTHONUNBUFFERED=1` is set for real-time logging
