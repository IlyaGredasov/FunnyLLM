from utils.extract import extract_anecdotes
from utils.extract import get_raw_pages


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2, help="Number of pages to fetch")
    args = parser.parse_args()

    pages = await get_raw_pages(args.pages)
    for i, p in enumerate(pages):
        for j, a in enumerate(extract_anecdotes(p)):
            print(f"{i + 1}.{j + 1}#####")
            print(a)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
