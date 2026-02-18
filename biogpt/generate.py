"""
Generate text from a trained BioGPT checkpoint.

Usage:
  python -m biogpt.generate --checkpoint checkpoints/biogpt_final.pt --prompt "The capital of France is"
  python -m biogpt.generate --checkpoint checkpoints/biogpt_final.pt --interactive
"""

import argparse
import torch
from biogpt.model import BioGPT


def load_model(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    config["use_checkpoint"] = False
    model = BioGPT(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded BioGPT: {n:,} params | val_loss={ckpt.get('val_loss', '?')}")
    return model


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40, device="cuda"):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        out = model.generate(ids, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    model = load_model(args.checkpoint, args.device)

    if args.interactive:
        print("\nInteractive mode (Ctrl+C to exit)")
        print("-" * 40)
        while True:
            try:
                prompt = input("\nPrompt> ")
                if not prompt.strip():
                    continue
                text = generate(model, tokenizer, prompt, args.max_tokens,
                                args.temperature, args.top_k, args.device)
                print(f"\n{text}")
            except KeyboardInterrupt:
                print("\nDone.")
                break
    elif args.prompt:
        text = generate(model, tokenizer, args.prompt, args.max_tokens,
                        args.temperature, args.top_k, args.device)
        print(text)
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
