from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import sentencepiece as spm
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from .network import FunnyLLM

load_dotenv()

DEFAULT_TOKENIZER_MODEL = Path(os.getenv("DEFAULT_TOKENIZER_MODEL", "sp_anecdote.model"))
DEFAULT_DATA_PATH = Path(os.getenv("DEFAULT_DATA_PATH", "anecdotes.parquet"))
DEFAULT_OUT_DIR = Path(os.getenv("DEFAULT_OUT_DIR", "checkpoints"))


def load_tokenizer(model_path: Path | str = DEFAULT_TOKENIZER_MODEL) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(model_path))


def tokenize(texts: list[str], model_path: Path | str = DEFAULT_TOKENIZER_MODEL) -> list[list[int]]:
    tokenizer = load_tokenizer(model_path)
    return [tokenizer.encode(t, out_type=int) for t in texts]


def iter_token_stream(tokenizer: spm.SentencePieceProcessor, texts: Iterable[str]) -> list[int]:
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    stream: list[int] = []
    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        if bos_id != -1:
            stream.append(bos_id)
        stream.extend(tokenizer.encode(cleaned, out_type=int))
        if eos_id != -1:
            stream.append(eos_id)
    return stream


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    max_start = data.size(0) - seq_len - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for the requested seq_len")
    index = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i: i + seq_len] for i in index]).to(device)
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in index]).to(device)
    return x, y


def generate_sample(model: FunnyLLM, tokenizer: spm.SentencePieceProcessor, prompt: str, device: str,
                    max_new_tokens: int) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, out_type=int)], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", default=str(DEFAULT_TOKENIZER_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Раз")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.data, columns=["anecdote"])
    texts = [t for t in table["anecdote"].to_pylist() if t]

    tokenizer = load_tokenizer(args.model)
    token_stream = iter_token_stream(tokenizer, texts)
    if not token_stream:
        raise ValueError("Token stream is empty")

    data = torch.tensor(token_stream, dtype=torch.long)
    if args.ckpt != "":
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        model = FunnyLLM(vocab_size=checkpoint["config"]["vocab_size"], max_seq_len=checkpoint["config"]["max_seq_len"],
                         d_model=checkpoint["config"]["d_model"],
                         n_heads=checkpoint["config"]["n_heads"], n_layers=checkpoint["config"]["n_layers"],
                         dropout=checkpoint["config"]["dropout"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
    else:
        model = FunnyLLM(vocab_size=tokenizer.get_piece_size(), max_seq_len=args.seq_len, d_model=args.d_model,
                         n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = max(1, data.size(0) // (args.batch_size * args.seq_len))

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", unit="step")
        total_loss = 0.0
        for _ in progress:
            x, y = get_batch(data, args.batch_size, args.seq_len, args.device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, steps_per_epoch)
        sample = generate_sample(model, tokenizer, args.prompt, args.device, args.max_new_tokens)
        print(f"epoch {epoch} avg loss: {avg_loss:.4f}")
        print("sample:")
        print(sample)

        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.get_piece_size(),
                    "max_seq_len": args.seq_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                },
                "tokenizer_model": str(args.model),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
