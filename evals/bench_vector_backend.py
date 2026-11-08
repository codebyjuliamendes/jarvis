#!/usr/bin/env python3
"""Microbenchmark da perna vetorial: sqlite-vec x NumPy sobre o JSON.

No corpus do eval (25 trechos) a latência da busca é dominada pelo embedding da
pergunta no Ollama (~0,1 s), então os dois backends empatam. Aqui o índice é
sintético (vetores aleatórios de 768 dimensões, como os do nomic-embed-text) e o
embedding da consulta é fixo: mede só a busca, para N trechos.

    .venv/Scripts/python.exe evals/bench_vector_backend.py --sizes 25,1000,4000
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common  # noqa: E402
from jarvis.metrics_utils import latency_summary  # noqa: E402
from jarvis.rag import RagIndex  # noqa: E402

DIM = 768


def build(root: Path, backend: str, size: int, seed: int) -> RagIndex:
    settings = replace(common.make_eval_settings(root), rag_vector_backend=backend)
    rag = RagIndex(settings)
    rng = random.Random(seed)
    per_doc = 20
    for start in range(0, size, per_doc):
        count = min(per_doc, size - start)
        chunks = ["trecho {} do documento".format(start + i) for i in range(count)]
        vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(count)]
        rag._store_document(str(root / "doc{}.md".format(start)), 0.0, 0, chunks, vectors)
    return rag


def bench(size: int, backend: str, queries: int, seed: int) -> dict:
    rng = random.Random(seed + 1)
    with tempfile.TemporaryDirectory() as tmp:
        rag = build(Path(tmp).resolve(), backend, size, seed)
        try:
            assert rag.vector_backend == backend, "sqlite-vec não carregou"
            times: List[float] = []
            for _ in range(queries):
                query = [rng.gauss(0, 1) for _ in range(DIM)]
                rag._embed_query = lambda _text, q=query: q  # type: ignore[assignment]
                started = time.perf_counter()
                hits = rag._vector_hits("x", 5)
                times.append(time.perf_counter() - started)
                assert hits and len(hits) == 5
        finally:
            rag.close()
    return latency_summary(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="25,1000,4000")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    args = parser.parse_args()
    print("| Trechos | NumPy (JSON) média (ms) | sqlite-vec média (ms) | Razão |")
    print("|---|---|---|---|")
    for size in [int(s) for s in args.sizes.split(",")]:
        numpy_s = bench(size, "numpy", args.queries, args.seed)["mean"]
        vec_s = bench(size, "sqlite-vec", args.queries, args.seed)["mean"]
        print("| {} | {} | {} | {}x |".format(
            size, common.fmt_num(numpy_s * 1000, 2), common.fmt_num(vec_s * 1000, 2),
            common.fmt_num(numpy_s / vec_s, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
