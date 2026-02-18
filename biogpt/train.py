"""
Training Script -- Train BioGPT on The Pile

Features:
  - Memory-mapped data loading (low RAM usage for large datasets)
  - Mixed precision (bfloat16 on CUDA)
  - Gradient accumulation
  - Cosine LR with warmup (matching Pythia: lr=6e-4, betas 0.9/0.95)
  - Periodic checkpointing with resume
  - Validation loss tracking

Usage:
  python -m biogpt.train --data_dir data --out_dir checkpoints
  python -m biogpt.train --data_dir data --out_dir checkpoints --timing_test
"""

import os
import sys
import time
import math
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from biogpt.model import BioGPT, create_biogpt, count_parameters


class MemmapDataLoader:
    """Efficient data loader reading from memory-mapped binary token files."""

    def __init__(self, data_path: str, block_size: int, batch_size: int,
                 device: str = "cpu"):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.n_tokens = len(self.data)

    def get_batch(self):
        ix = torch.randint(self.n_tokens - self.block_size - 1,
                           (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(self.device), y.to(self.device)

    def tokens_per_batch(self):
        return self.batch_size * self.block_size


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps=50):
    model.eval()
    total_loss = 0.0
    for _ in range(eval_steps):
        x, y = val_loader.get_batch()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / eval_steps


def train(args):
    device = "cuda"
    print(f"Device: {device}")

    train_loader = MemmapDataLoader(
        os.path.join(args.data_dir, "train.bin"),
        block_size=args.block_size, batch_size=args.batch_size, device=device)
    val_loader = MemmapDataLoader(
        os.path.join(args.data_dir, "val.bin"),
        block_size=args.block_size, batch_size=args.batch_size, device=device)

    print(f"Train tokens: {train_loader.n_tokens:,}")
    print(f"Val tokens:   {val_loader.n_tokens:,}")

    tokens_per_step = train_loader.tokens_per_batch() * args.grad_accum
    total_steps = args.target_tokens // tokens_per_step
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total steps: {total_steps:,}")

    model = create_biogpt(
        soma_ffn_expansion=args.soma_ffn_expansion,
        n_passes=args.n_passes,
        dropout=0.0,
        max_seq_len=args.block_size,
        use_checkpoint=True,
    )
    n_params = count_parameters(model)
    print(f"Model params: {n_params:,}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_step = 0
    os.makedirs(args.out_dir, exist_ok=True)
    resume_path = os.path.join(args.out_dir, "biogpt_latest.pt")
    if os.path.exists(resume_path) and args.resume:
        print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  Resumed at step {start_step}")

    if args.timing_test:
        print(f"\n{'='*60}")
        print(f"  TIMING TEST: {args.timing_steps} steps")
        print(f"{'='*60}")
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for step in range(args.timing_steps):
            optimizer.zero_grad()
            for micro in range(args.grad_accum):
                x, y = train_loader.get_batch()
                with torch.amp.autocast(device_type="cuda",
                                         dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step % 10 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                tps = (step + 1) * tokens_per_step / elapsed
                print(f"  Step {step:4d} | loss={loss.item()*args.grad_accum:.4f} | "
                      f"{tps:.0f} tok/s | "
                      f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        torch.cuda.synchronize()
        total_time = time.time() - t0
        total_tokens = args.timing_steps * tokens_per_step
        tps = total_tokens / total_time
        time_for_2B = 2_000_000_000 / tps / 3600

        print(f"\n  --- TIMING RESULTS ---")
        print(f"  Throughput: {tps:.0f} tok/s")
        print(f"  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        print(f"  Estimated time for 2B tokens: {time_for_2B:.1f} hours")
        return

    print(f"\n{'='*60}")
    print(f"  TRAINING: BioGPT | {n_params:,} params")
    print(f"  Target: {args.target_tokens:,} tokens ({total_steps} steps)")
    print(f"{'='*60}\n")

    model.train()
    best_val_loss = float('inf')
    log_data = []
    t0 = time.time()
    tokens_processed = start_step * tokens_per_step

    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup_steps, total_steps,
                    args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(args.grad_accum):
            x, y = train_loader.get_batch()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step

        if step % args.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_per_step * args.log_interval / dt if step > start_step else 0
            print(f"  Step {step:>6d}/{total_steps} | "
                  f"loss={accum_loss:.4f} | lr={lr:.2e} | "
                  f"{tps:.0f} tok/s | "
                  f"{tokens_processed/1e9:.2f}B tokens")
            t0 = time.time()

        if step > 0 and step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader)
            print(f"  >>> Val loss: {val_loss:.4f}")

            log_entry = {
                "step": step, "train_loss": accum_loss,
                "val_loss": val_loss, "lr": lr,
                "tokens": tokens_processed,
            }
            log_data.append(log_entry)

            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": model.config,
                "tokens_processed": tokens_processed,
            }
            torch.save(ckpt, resume_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, "biogpt_best.pt")
                torch.save(ckpt, best_path)
                print(f"  >>> New best! Saved to {best_path}")

            with open(os.path.join(args.out_dir, "biogpt_log.json"), "w") as f:
                json.dump(log_data, f, indent=2)

            model.train()

    val_loss = estimate_loss(model, val_loader)
    print(f"\n  Final val loss: {val_loss:.4f}")

    ckpt = {
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": model.config,
        "tokens_processed": tokens_processed,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "biogpt_final.pt"))
    print(f"  Saved final checkpoint.")

    with open(os.path.join(args.out_dir, "biogpt_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--soma_ffn_expansion", type=float, default=3.0)
    parser.add_argument("--n_passes", type=int, default=1)
    parser.add_argument("--timing_test", action="store_true")
    parser.add_argument("--timing_steps", type=int, default=50)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
