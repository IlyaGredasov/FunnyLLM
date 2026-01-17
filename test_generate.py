import argparse

import torch

from model.network import FunnyLLM
from model.train import DEFAULT_TOKENIZER_MODEL, generate_sample, load_tokenizer


def _prompt_int(message: str, default: int) -> int:
    raw = input(message).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid number, using default {default}.")
        return default
    if value <= 0:
        print(f"Number must be > 0, using default {default}.")
        return default
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_MODEL))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    cfg = checkpoint["config"]
    model = FunnyLLM(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = load_tokenizer(args.tokenizer)

    while True:
        prompt = input("Prompt (default: 'Раз', 'exit' to quit): ").strip()
        if not prompt:
            prompt = "Раз"
        if prompt.lower() in {"exit", "quit"}:
            break
        max_new_tokens = _prompt_int("Max new tokens (default: 80): ", 80)
        sample = generate_sample(
            model,
            tokenizer,
            prompt,
            args.device,
            max_new_tokens,
        )
        print(sample)


if __name__ == "__main__":
    main()
