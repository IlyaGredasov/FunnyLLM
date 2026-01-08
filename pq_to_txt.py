import pyarrow.parquet as pq

table = pq.read_table("anecdotes.parquet", columns=["anecdote"])
with open("anecdotes.txt", "w", encoding="utf-8") as f:
    for a in table["anecdote"].to_pylist():
        a = (a or "").strip()
        if a:
            f.write(a.replace("\n", " ") + "\n")
