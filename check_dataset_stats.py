import argparse

import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="anecdotes.parquet", help="Path to parquet dataset")
    parser.add_argument("--model", required=True, help="Path to SentencePiece model file (.model)")
    parser.add_argument("--nbins", type=int, default=10, help="Number of bins for length distribution")
    args = parser.parse_args()

    if args.nbins <= 0:
        raise ValueError("--nbins must be > 0")

    table = pq.read_table(args.data, columns=["anecdote"])
    anecdotes = table["anecdote"].to_pylist()

    tokenizer = spm.SentencePieceProcessor(model_file=args.model)

    lengths: list[int] = []
    total_tokens = 0
    for text in tqdm(anecdotes, desc="Tokenizing", unit="anekdot"):
        tokens = tokenizer.encode(text, out_type=int)
        length = len(tokens)
        lengths.append(length)
        total_tokens += length

    if not lengths:
        print("No anecdotes found.")
        return

    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = total_tokens / len(lengths)

    bin_size = max(1, (max_len + args.nbins - 1) // args.nbins)
    bins = [0] * args.nbins
    for length in lengths:
        index = min(length // bin_size, args.nbins - 1)
        bins[index] += 1

    print(f"total anecdotes: {len(lengths)}")
    print(f"total tokens: {total_tokens}")
    print(f"min tokens: {min_len}")
    print(f"max tokens: {max_len}")
    print(f"avg tokens: {avg_len:.2f}")
    print("distribution:")
    for i, count in enumerate(bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, max_len)
        print(f"{start}-{end}: {count}")


if __name__ == "__main__":
    main()
