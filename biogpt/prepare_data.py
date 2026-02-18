"""
Prepare training data from The Pile -- same dataset Pythia was trained on.

Streams from HuggingFace, tokenizes with the GPT-NeoX-20B tokenizer
(same as Pythia), and writes directly to memory-mapped files (low RAM usage).

Includes retry logic for unreliable streaming connections,
with progress saving and resume capability.

Usage:
  python -m biogpt.prepare_data --target_tokens 2000000000 --out_dir data
  python -m biogpt.prepare_data --target_tokens 100000000 --out_dir data  # quick test
"""

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm


def create_dataset(source_idx=0):
    """Create a streaming dataset, trying multiple sources."""
    from datasets import load_dataset

    pile_sources = [
        ("monology/pile-uncopyrighted", None),
        ("EleutherAI/pile", "all"),
        ("Skylion007/openwebtext", None),
    ]

    for i in range(source_idx, len(pile_sources)):
        source, config = pile_sources[i]
        try:
            print(f"  Trying {source}...")
            kwargs = dict(split="train", streaming=True)
            if config:
                dataset = load_dataset(source, config, **kwargs)
            else:
                dataset = load_dataset(source, **kwargs)
            sample = next(iter(dataset))
            text_field = "text" if "text" in sample else list(sample.keys())[0]
            print(f"  Success! Text field: '{text_field}'")
            return dataset, text_field, i
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError("All data sources failed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--max_retries", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    print("Loading GPT-NeoX-20B tokenizer (same as Pythia)...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    eos_token = tokenizer.eos_token_id
    print(f"  Vocab size: {tokenizer.vocab_size}, EOS: {eos_token}")

    total_target = args.target_tokens
    n_val = int(total_target * args.val_fraction)
    n_train = total_target - n_val

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    offset_file = os.path.join(args.out_dir, "offset.txt")
    train_offset, val_offset, docs_skipped = 0, 0, 0

    if os.path.exists(offset_file):
        with open(offset_file, "r") as f:
            parts = f.read().strip().split(",")
            train_offset, val_offset, docs_skipped = int(parts[0]), int(parts[1]), int(parts[2])
        print(f"  Resuming: train={train_offset:,}, val={val_offset:,}, skip {docs_skipped:,} docs")

    print(f"\nPre-allocating memmap files...")
    print(f"  train: {n_train:,} tokens ({n_train * 2 / 1e9:.2f} GB)")
    print(f"  val:   {n_val:,} tokens ({n_val * 2 / 1e6:.1f} MB)")

    train_mm = np.memmap(train_path, dtype=np.uint16,
                          mode='r+' if train_offset > 0 else 'w+', shape=(n_train,))
    val_mm = np.memmap(val_path, dtype=np.uint16,
                        mode='r+' if val_offset > 0 else 'w+', shape=(n_val,))

    source_idx = 0
    filling_val = train_offset >= n_train
    docs_processed = docs_skipped
    retries = 0

    already_done = train_offset + val_offset
    pbar = tqdm(total=total_target, initial=already_done, unit="tok", unit_scale=True)

    CHUNK_SIZE = 500_000
    chunk_buffer = []
    chunk_len = 0
    save_interval = 10_000_000
    last_save = already_done

    def flush_buffer():
        nonlocal train_offset, val_offset, filling_val, chunk_buffer, chunk_len
        if chunk_len == 0:
            return
        tokens_array = np.concatenate(chunk_buffer).astype(np.uint16)

        if not filling_val:
            space = n_train - train_offset
            to_write = min(len(tokens_array), space)
            train_mm[train_offset:train_offset + to_write] = tokens_array[:to_write]
            train_offset += to_write
            if to_write < len(tokens_array):
                filling_val = True
                leftover = tokens_array[to_write:]
                val_to_write = min(len(leftover), n_val - val_offset)
                val_mm[val_offset:val_offset + val_to_write] = leftover[:val_to_write]
                val_offset += val_to_write
        else:
            to_write = min(len(tokens_array), n_val - val_offset)
            val_mm[val_offset:val_offset + to_write] = tokens_array[:to_write]
            val_offset += to_write

        chunk_buffer, chunk_len = [], 0

    def save_progress():
        nonlocal last_save
        train_mm.flush()
        val_mm.flush()
        with open(offset_file, "w") as f:
            f.write(f"{train_offset},{val_offset},{docs_processed}")
        last_save = train_offset + val_offset

    while retries < args.max_retries:
        try:
            dataset, text_field, source_idx = create_dataset(source_idx)
            doc_count = 0
            for doc in dataset:
                doc_count += 1
                if doc_count <= docs_skipped:
                    continue

                text = doc[text_field]
                if not text or len(text.strip()) < 50:
                    docs_processed += 1
                    continue

                tokens = tokenizer.encode(text)
                if eos_token is not None:
                    tokens.append(eos_token)

                chunk_buffer.append(np.array(tokens, dtype=np.uint16))
                chunk_len += len(tokens)
                docs_processed += 1
                pbar.update(len(tokens))

                if chunk_len >= CHUNK_SIZE:
                    flush_buffer()

                current_total = train_offset + val_offset + chunk_len
                if current_total - last_save >= save_interval:
                    flush_buffer()
                    save_progress()

                if train_offset + val_offset + chunk_len >= total_target:
                    break

            flush_buffer()
            save_progress()
            break

        except KeyboardInterrupt:
            flush_buffer()
            save_progress()
            sys.exit(1)

        except Exception as e:
            retries += 1
            flush_buffer()
            save_progress()
            docs_skipped = docs_processed
            print(f"\n  Stream error (retry {retries}): {e}")
            if retries < args.max_retries:
                time.sleep(min(30, 2 ** retries))

    pbar.close()

    if os.path.exists(offset_file):
        os.remove(offset_file)

    print(f"\nDone! {docs_processed:,} documents")
    print(f"  train: {train_offset:,} tokens | val: {val_offset:,} tokens")


if __name__ == "__main__":
    main()
