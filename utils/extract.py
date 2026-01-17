import asyncio
import html as html_lib
import os
import re
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()


async def get_raw_pages(
    n: int, url: str = os.getenv("DEFAULT_URL"), timeout_s: int = 10
) -> List[str]:
    if n <= 0:
        return []

    async with httpx.AsyncClient(
        timeout=timeout_s, headers={"User-Agent": "FunnyLLM/1.0"}
    ) as client:
        tasks = [client.get(url) for _ in range(n)]
        responses = await asyncio.gather(*tasks)

    for response in responses:
        response.raise_for_status()

    return [response.text for response in responses]


def extract_anecdotes(html_text: str) -> List[str]:
    anecdotes: List[str] = []
    parts = html_text.split('<div class="topicbox"')

    for part in parts[1:]:
        header, _, rest = part.partition(">")
        if 'data-t="j"' not in header:
            continue

        match = re.search(r'<div class="text">(.*?)</div>', rest, re.S)
        if not match:
            continue

        raw = match.group(1)
        raw = re.sub(r"<br\s*/?>", "\n", raw)
        raw = re.sub(r"<[^>]+>", "", raw)
        cleaned = html_lib.unescape(raw).strip()
        if cleaned:
            anecdotes.append(cleaned)

    return anecdotes
