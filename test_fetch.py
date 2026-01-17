import argparse
import asyncio
import os

from tqdm import tqdm

from utils.extract import extract_anecdotes, get_raw_pages
from utils.fetch import build_and_save_parquet, load_dataset_parquet


async def run(pages: int, output: str, delay_s: float) -> None:
    delay_s = max(0.0, delay_s)
    existing_anecdotes: list[str] = []
    if os.path.exists(output):
        existing_table = load_dataset_parquet(output)
        if "anecdote" in existing_table.column_names:
            existing_anecdotes = existing_table["anecdote"].to_pylist()

    seen: set[str] = {a.strip() for a in existing_anecdotes if a.strip()}
    new_anecdotes: list[str] = []
    total_count = len(seen)

    progress = tqdm(range(pages), desc="Fetching pages", unit="page")
    for _ in progress:
        raw_pages = await get_raw_pages(1)
        for page in raw_pages:
            for anecdote in extract_anecdotes(page):
                cleaned = anecdote.strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                new_anecdotes.append(cleaned)
                total_count += 1
        progress.set_postfix(total=total_count)
        if delay_s:
            await asyncio.sleep(delay_s)

    table = build_and_save_parquet(new_anecdotes, output)
    print(f"Saved {table.num_rows} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, required=True, help="Number of pages")
    parser.add_argument(
        "--out",
        default="anecdotes.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Delay between starting requests in seconds",
    )
    args = parser.parse_args()

    asyncio.run(run(args.pages, args.out, args.timeout))


if __name__ == "__main__":
    main()
