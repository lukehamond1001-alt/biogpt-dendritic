"""
BioGPT vs Pythia-160M -- Head-to-Head Comparison

Runs a full comparison pipeline:
  1. Generation comparison (same prompts, side by side)
  2. Validation loss comparison
  3. Pruning curve: magnitude pruning + dead neuron removal
  4. Fine-tune after each pruning level, re-evaluate

Usage:
  python -m biogpt.compare --data_dir data --biogpt_ckpt checkpoints/biogpt_final.pt
"""

import os, sys, copy, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROMPTS = [
    {"id": "know1", "cat": "knowledge", "prompt": "The capital of France is", "max_tok": 40},
    {"id": "know2", "cat": "knowledge", "prompt": "Water boils at a temperature of", "max_tok": 40},
    {"id": "know3", "cat": "knowledge", "prompt": "The theory of relativity was developed by", "max_tok": 50},
    {"id": "reason1", "cat": "reasoning", "prompt": "If all cats are animals, and all animals need water, then all cats", "max_tok": 40},
    {"id": "code1", "cat": "code", "prompt": "def fibonacci(n):\n    \"\"\"Return the nth fibonacci number.\"\"\"\n", "max_tok": 80},
    {"id": "code2", "cat": "code", "prompt": "# Python function to sort a list\ndef sort_list(lst):\n", "max_tok": 60},
    {"id": "sci1", "cat": "science", "prompt": "Photosynthesis is the process by which plants", "max_tok": 60},
    {"id": "sci2", "cat": "science", "prompt": "The human brain contains approximately", "max_tok": 50},
    {"id": "math1", "cat": "math", "prompt": "To calculate the area of a circle, you use the formula", "max_tok": 50},
    {"id": "write1", "cat": "writing", "prompt": "Once upon a time, in a kingdom far away, there lived a", "max_tok": 80},
    {"id": "inst1", "cat": "instruction", "prompt": "Question: What is the largest planet in our solar system?\nAnswer:", "max_tok": 40},
    {"id": "long1", "cat": "coherence", "prompt": "In machine learning, neural networks are inspired by biological neurons. The key idea is that", "max_tok": 120},
]


def load_biogpt(checkpoint_path, device="cuda"):
    from biogpt.model import BioGPT
    print(f"Loading BioGPT from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    config["use_checkpoint"] = False
    model = BioGPT(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  BioGPT: {n:,} params | val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model, ckpt


def load_pythia(step=1000, device="cuda"):
    from transformers import GPTNeoXForCausalLM
    print(f"Loading Pythia-160M (step{step})...")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m", revision=f"step{step}",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Pythia-160M: {n:,} params")
    return model


@torch.no_grad()
def generate_biogpt(model, tokenizer, prompt, max_tokens, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, top_k=40)
    return tokenizer.decode(out[0, ids.shape[1]:].tolist())


@torch.no_grad()
def generate_pythia(model, tokenizer, prompt, max_tokens, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7,
                             top_k=40, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:].tolist())


@torch.no_grad()
def eval_val_loss(model, data_path, block_size, device, is_pythia=False,
                  n_batches=50, batch_size=4):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_pythia:
                loss = model(x, labels=x).loss
            else:
                _, loss = model(x, y)
        total += loss.item()
    return total / n_batches


def magnitude_prune(model, sparsity):
    """Prune smallest weights by magnitude. Returns (actual_sparsity, mask_dict)."""
    if sparsity <= 0:
        return 0.0, {}

    skip = ['embed', 'emb', 'norm', 'ln', 'nmda', 'layernorm', 'pos_emb',
            'token_emb', 'pass_weight', 'neuron_weight']

    all_weights = []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and not any(s in name.lower() for s in skip):
            all_weights.append(p.data.abs().flatten())

    all_mag = torch.cat(all_weights)
    k = int(sparsity * all_mag.numel())
    if k == 0:
        return 0.0, {}
    threshold = torch.kthvalue(all_mag, k).values.item()

    total, pruned = 0, 0
    mask_dict = {}
    for name, p in model.named_parameters():
        if p.dim() >= 2 and not any(s in name.lower() for s in skip):
            mask = (p.data.abs() > threshold).float()
            p.data *= mask
            mask_dict[name] = mask
            total += p.numel()
            pruned += (mask == 0).sum().item()

    return (pruned / total if total > 0 else 0.0), mask_dict


def kill_dead_neurons(model, threshold=0.95):
    """If a neuron has >threshold fraction of weights zeroed, kill all remaining.
    Simulates biological apoptosis -- a mostly-disconnected neuron is useless."""
    killed = 0
    try:
        for layer in model.layers:
            for neuron in layer.neurons:
                total, zeros = 0, 0
                for p in neuron.parameters():
                    total += p.numel()
                    zeros += (p.data == 0).sum().item()
                if total > 0 and zeros / total > threshold:
                    for p in neuron.parameters():
                        p.data.zero_()
                    killed += 1
    except AttributeError:
        pass
    if killed > 0:
        print(f"    Dead neuron cleanup: {killed} neurons killed (>{threshold*100:.0f}% zeroed)")
    return killed


def finetune(model, data_path, block_size, device, is_pythia=False,
             steps=500, batch_size=4, lr=1e-4, prune_mask=None):
    """Fine-tune with optional pruning mask to keep zeroed weights at zero."""
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_pythia:
                loss = model(x, labels=x).loss
            else:
                _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if prune_mask is not None:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in prune_mask:
                        p.data *= prune_mask[name]
        if step % 100 == 0:
            print(f"      ft {step}/{steps} loss={loss.item():.4f}")
    model.eval()


def count_nonzero_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p.data != 0).sum().item()
    return nonzero, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--biogpt_ckpt", default="checkpoints/biogpt_final.pt")
    parser.add_argument("--pythia_step", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--finetune_steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    val_path = os.path.join(args.data_dir, "val.bin")
    train_path = os.path.join(args.data_dir, "train.bin")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: BioGPT vs Pythia-160M")
    print("=" * 70)

    biogpt, _ = load_biogpt(args.biogpt_ckpt, args.device)
    pythia = load_pythia(args.pythia_step, args.device)

    all_gens = []
    for p in PROMPTS:
        print(f"\n  [{p['cat']:>12s}] {p['id']}")
        print(f"  PROMPT: {p['prompt'][:80]}")
        bio = generate_biogpt(biogpt, tokenizer, p["prompt"], p["max_tok"], args.device)
        pyt = generate_pythia(pythia, tokenizer, p["prompt"], p["max_tok"], args.device)
        print(f"  BIOGPT: {bio[:120].strip()}")
        print(f"  PYTHIA: {pyt[:120].strip()}")
        all_gens.append({"id": p["id"], "cat": p["cat"], "prompt": p["prompt"],
                         "biogpt": bio, "pythia": pyt})

    print("\n" + "=" * 70)
    print("  VALIDATION LOSS")
    print("=" * 70)

    bio_val = eval_val_loss(biogpt, val_path, 1024, args.device)
    pyth_val = eval_val_loss(pythia, val_path, 1024, args.device, is_pythia=True)
    print(f"  BioGPT (294M):  val_loss={bio_val:.4f}  ppl={np.exp(bio_val):.2f}")
    print(f"  Pythia (162M):  val_loss={pyth_val:.4f}  ppl={np.exp(pyth_val):.2f}")

    del biogpt, pythia
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  BIOGPT PRUNING CURVE (magnitude + dead neuron removal)")
    print("=" * 70)

    biogpt_base, _ = load_biogpt(args.biogpt_ckpt, "cpu")
    bio_results = []

    for sparsity in [0.0, 0.25, 0.45, 0.70, 0.90]:
        print(f"\n  --- Sparsity: {sparsity*100:.0f}% ---")
        model = copy.deepcopy(biogpt_base).to(args.device)

        if sparsity > 0:
            actual, mask = magnitude_prune(model, sparsity)
            killed = kill_dead_neurons(model)
            print(f"    Pruned {actual*100:.1f}%")
            finetune(model, train_path, 1024, args.device,
                     steps=args.finetune_steps, prune_mask=mask)
        else:
            actual = 0.0

        nonzero, total = count_nonzero_params(model)
        val_loss = eval_val_loss(model, val_path, 1024, args.device)
        ppl = np.exp(val_loss)
        print(f"    Non-zero: {nonzero:,} / {total:,} | loss={val_loss:.4f} ppl={ppl:.2f}")

        bio_results.append({"sparsity": sparsity, "nonzero_params": nonzero,
                            "val_loss": val_loss, "perplexity": ppl})
        del model; torch.cuda.empty_cache()

    del biogpt_base; torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  PYTHIA PRUNING CURVE (magnitude)")
    print("=" * 70)

    pythia_base = load_pythia(args.pythia_step, "cpu")
    pyth_results = []

    for sparsity in [0.0, 0.25, 0.50, 0.70, 0.90]:
        print(f"\n  --- Sparsity: {sparsity*100:.0f}% ---")
        model = copy.deepcopy(pythia_base).to(args.device)

        if sparsity > 0:
            actual, mask = magnitude_prune(model, sparsity)
            finetune(model, train_path, 1024, args.device, is_pythia=True,
                     steps=args.finetune_steps, prune_mask=mask)
        else:
            actual = 0.0

        nonzero, total = count_nonzero_params(model)
        val_loss = eval_val_loss(model, val_path, 1024, args.device, is_pythia=True)
        ppl = np.exp(val_loss)
        print(f"    Non-zero: {nonzero:,} / {total:,} | loss={val_loss:.4f} ppl={ppl:.2f}")

        pyth_results.append({"sparsity": sparsity, "nonzero_params": nonzero,
                             "val_loss": val_loss, "perplexity": ppl})
        del model; torch.cuda.empty_cache()

    results = {"biogpt_base_val": bio_val, "pythia_base_val": pyth_val,
               "biogpt_prune": bio_results, "pythia_prune": pyth_results,
               "generations": all_gens}
    with open(os.path.join(args.output_dir, "full_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to {args.output_dir}/full_comparison.json")


if __name__ == "__main__":
    main()
