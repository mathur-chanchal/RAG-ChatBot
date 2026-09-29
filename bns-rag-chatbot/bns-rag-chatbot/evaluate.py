"""Measure retrieval quality and latency so the numbers you quote are your own.

Create eval_set.json, a list of questions with the section number(s) that should be retrieved:
    [{"question": "What is the punishment for murder?", "expected_sections": ["103"]}, ...]

Run:
    python evaluate.py --k 4                       # retrieval hit-rate + retrieval latency
    python evaluate.py --k 4 --with-llm            # also end-to-end answer latency
    python evaluate.py --compare-chunks 500 1000   # compare chunk sizes (re-indexes to a temp dir)
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import config


def hit(docs, expected):
    text = " ".join(d.page_content for d in docs) + " " + " ".join(d.metadata.get("section", "") for d in docs)
    return any(f"{s}." in text or f"Section {s}" in text for s in expected)


def eval_store(store, items, k):
    hits, times = 0, []
    for it in items:
        t0 = time.time()
        docs = store.max_marginal_relevance_search(it["question"], k=k, fetch_k=k * 4)
        times.append(time.time() - t0)
        hits += hit(docs, it["expected_sections"])
    return hits / len(items), statistics.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--compare-chunks", type=int, nargs="+")
    args = ap.parse_args()
    items = json.loads(Path(args.eval_set).read_text())

    if args.compare_chunks:
        import ingest

        pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
        emb = ingest.get_embeddings()
        for size in args.compare_chunks:
            config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, size // 7
            with tempfile.TemporaryDirectory() as tmp:
                store = ingest.build_index(pdfs, embeddings=emb, persist_dir=tmp)
                acc, lat = eval_store(store, items, args.k)
                n = store._collection.count()
            print(f"chunk_size={size:>5}  chunks={n:>5}  hit@{args.k}={acc:.0%}  retrieval={lat*1000:.0f}ms")
        return

    from rag import BNSAssistant

    bot = BNSAssistant(top_k=args.k) if args.with_llm else None
    if bot:
        store = bot.store
    else:
        from langchain_chroma import Chroma
        from ingest import get_embeddings

        store = Chroma(collection_name=config.COLLECTION_NAME, embedding_function=get_embeddings(),
                       persist_directory=config.CHROMA_DIR)
    acc, lat = eval_store(store, items, args.k)
    print(f"Questions: {len(items)} | hit@{args.k}: {acc:.0%} | mean retrieval latency: {lat*1000:.0f} ms")
    print(f"Indexed chunks: {store._collection.count()}")

    if bot:
        times = [bot.ask(it["question"]).latency_s for it in items]
        print(f"End-to-end latency: mean {statistics.mean(times):.1f}s, "
              f"p95 {sorted(times)[int(0.95 * (len(times) - 1))]:.1f}s")


if __name__ == "__main__":
    main()
