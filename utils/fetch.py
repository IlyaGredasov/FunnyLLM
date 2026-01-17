from __future__ import annotations

import os
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def build_dataset(
    anecdotes: Iterable[str], existing: Iterable[str] | None = None
) -> pa.Table:
    unique: list[str] = []
    seen: set[str] = set()

    if existing:
        for anecdote in existing:
            cleaned = anecdote.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)

    for anecdote in anecdotes:
        cleaned = anecdote.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)

    ids = list(range(len(unique)))
    return pa.table({"id": ids, "anecdote": unique})


def save_dataset_parquet(table: pa.Table, path: str) -> None:
    pq.write_table(table, path)


def load_dataset_parquet(path: str) -> pa.Table:
    return pq.read_table(path)


def build_and_save_parquet(anecdotes: Iterable[str], path: str) -> pa.Table:
    existing_anecdotes: list[str] = []
    if os.path.exists(path):
        existing_table = load_dataset_parquet(path)
        if "anecdote" in existing_table.column_names:
            existing_anecdotes = existing_table["anecdote"].to_pylist()

    table = build_dataset(anecdotes, existing=existing_anecdotes)
    save_dataset_parquet(table, path)
    return table
