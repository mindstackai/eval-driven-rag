"""
EvalTrace — experiment logger and Phase 1 retrieval eval runner.

This module is the single source of truth for:
- Running Phase 1 retrieval eval (hit_rate@K, MRR, recall@K)
- Logging every experiment run to traces/experiment_log.jsonl
- Loading all past experiments as a flat pandas DataFrame for the dashboard

Design rules:
- Each eval run creates its own LanceDB table (overwrite=True) so stale
  vectors never pollute results.
- Table naming: f"eval_{embedder.model_name.replace('/', '_')}"
  → completely separate from training tables (corpus_*) in mine_negatives.py
- cosine metric on all tables → sim = 1 - _distance, always in [0, 1]
- Phase 2 LLM-as-judge eval lives in phase2_generation.py — not here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    from src.embedders.base import BaseEmbedder

# Default path for the append-only experiment log
_TRACE_PATH = os.path.join(
    os.path.dirname(__file__),  # src/eval/
    "..", "..",                  # project root
    "traces",
    "experiment_log.jsonl",
)
_TRACE_PATH = os.path.normpath(_TRACE_PATH)

_LANCEDB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "lancedb"
)
_LANCEDB_PATH = os.path.normpath(_LANCEDB_PATH)


# ---------------------------------------------------------------------------
# Phase 1 eval
# ---------------------------------------------------------------------------

def run_phase1_eval(
    embedder: "BaseEmbedder",
    chunks: list[str],
    eval_queries: list[dict],
    chunking_strategy: str,
    chunk_size: int,
    top_k: int = 5,
    lancedb_path: str | None = None,
    notes: str = "",
) -> dict:
    """
    Run Phase 1 retrieval eval for one (embedder × chunking config) pair.

    Args:
        embedder:          Any BaseEmbedder instance (HF or OpenAI).
        chunks:            List of plain text chunks to index.
        eval_queries:      List of dicts, each with:
                             - "question": str
                             - "relevant_chunk_ids": list[int]  ← positions in *chunks*
        chunking_strategy: Label for the chunking method used (e.g. "recursive").
        chunk_size:        Chunk size in tokens / characters (for logging).
        top_k:             Retrieval cut-off for hit_rate@K and recall@K.
        lancedb_path:      Override the LanceDB directory (default: ./lancedb).
        notes:             Free-text note appended to the experiment log entry.

    Returns:
        Dict with keys: hit_rate, mrr, recall_at_k, index_time_s, query_time_s,
        plus the full config and per-query details.
    """
    import lancedb

    db_path = lancedb_path or _LANCEDB_PATH
    os.makedirs(db_path, exist_ok=True)

    # Table name is unique per embedder — never shared with training tables
    safe_name = embedder.model_name.replace("/", "_").replace(".", "_").replace("-", "_")
    table_name = f"eval_{safe_name}"

    # ------------------------------------------------------------------
    # Index: encode corpus + write to LanceDB
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    corpus_vectors: np.ndarray = embedder.embed(chunks)  # (N, dim)

    rows = [
        {
            "chunk_id": idx,
            "text": text,
            "vector": corpus_vectors[idx].tolist(),
        }
        for idx, text in enumerate(chunks)
    ]

    db = lancedb.connect(db_path)
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.int32()),
            pa.field("text", pa.string()),
            pa.field(
                "vector",
                pa.list_(pa.float32(), embedder.embedding_dim),
            ),
        ]
    )
    # overwrite=True → each eval run gets a clean index
    table = db.create_table(
        table_name,
        data=rows,
        schema=schema,
        mode="overwrite",
    )

    index_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Query: encode each eval query + retrieve top-K
    # ------------------------------------------------------------------
    hit_rates: list[float] = []
    mrr_scores: list[float] = []
    recall_scores: list[float] = []
    per_query: list[dict] = []

    t1 = time.perf_counter()

    for item in eval_queries:
        question: str = item["question"]
        relevant_ids: list[int] = item.get("relevant_chunk_ids", [])

        # Encode single query (embed() expects a list)
        q_vec: np.ndarray = embedder.embed([question])[0]

        results = (
            table.search(q_vec.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        retrieved_ids = [r["chunk_id"] for r in results]
        # cosine metric → _distance in [0, 1] → sim = 1 - distance
        similarities = [round(1.0 - r["_distance"], 4) for r in results]

        relevant_set = set(relevant_ids)

        # Hit rate@K — 1 if any relevant id appears in top-K
        hit = 1.0 if any(rid in relevant_set for rid in retrieved_ids) else 0.0

        # MRR — reciprocal rank of first relevant result
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_set:
                mrr = 1.0 / rank
                break

        # Recall@K — fraction of relevant ids found in top-K
        if relevant_ids:
            found = sum(1 for rid in retrieved_ids if rid in relevant_set)
            recall = found / len(relevant_ids)
        else:
            recall = 0.0

        hit_rates.append(hit)
        mrr_scores.append(mrr)
        recall_scores.append(recall)

        per_query.append(
            {
                "question": question,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "similarities": similarities,
                "hit": hit,
                "mrr": mrr,
                "recall_at_k": recall,
            }
        )

    query_time_s = time.perf_counter() - t1

    n = len(eval_queries)
    metrics = {
        f"hit_rate@{top_k}": float(np.mean(hit_rates)) if n else 0.0,
        "mrr": float(np.mean(mrr_scores)) if n else 0.0,
        f"recall@{top_k}": float(np.mean(recall_scores)) if n else 0.0,
        "index_time_s": round(index_time_s, 3),
        "query_time_s": round(query_time_s, 3),
    }

    # ------------------------------------------------------------------
    # Log to traces/experiment_log.jsonl
    # ------------------------------------------------------------------
    log_experiment(
        embedder=embedder,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        top_k=top_k,
        metrics=metrics,
        notes=notes,
    )

    return {**metrics, "per_query": per_query}


# ---------------------------------------------------------------------------
# Experiment logger
# ---------------------------------------------------------------------------

def log_experiment(
    embedder: "BaseEmbedder",
    chunking_strategy: str,
    chunk_size: int,
    top_k: int,
    metrics: dict,
    notes: str = "",
    trace_path: str | None = None,
) -> dict:
    """
    Append one experiment record to traces/experiment_log.jsonl.

    Args:
        embedder:          The embedder used in this run.
        chunking_strategy: Chunking method label.
        chunk_size:        Chunk size used.
        top_k:             Retrieval cut-off.
        metrics:           Dict returned by run_phase1_eval (hit_rate, mrr, ...).
        notes:             Optional free-text annotation.
        trace_path:        Override log file path (for testing).

    Returns:
        The experiment record dict that was written.
    """
    path = trace_path or _TRACE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Deterministic run_id so identical configs are recognisable
    run_key = f"{embedder.model_name}|{chunking_strategy}|{chunk_size}"
    run_id = hashlib.md5(run_key.encode()).hexdigest()[:8]

    # Resolve base_model for fine-tuned variants (attribute is optional)
    base_model = getattr(embedder, "base_model", None)

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.embedding_dim,
            "source": embedder.source,
            "base_model": base_model,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "normalized": True,
            "cost_per_1k_tokens": embedder.cost_per_1k_tokens,
        },
        "metrics": metrics,
        "notes": notes,
    }

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


# ---------------------------------------------------------------------------
# Result loader — used by the dashboard
# ---------------------------------------------------------------------------

def load_experiments(trace_path: str | None = None) -> pd.DataFrame:
    """
    Read all experiment records from experiment_log.jsonl and return a
    flat DataFrame sorted by MRR descending.

    Config fields are flattened to columns with a ``cfg_`` prefix;
    metric fields keep their original names.

    Returns an empty DataFrame (with no columns) if the log file does not
    exist or is empty.
    """
    path = trace_path or _TRACE_PATH

    if not os.path.exists(path):
        return pd.DataFrame()

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        flat: dict = {
            "run_id": rec.get("run_id"),
            "timestamp": rec.get("timestamp"),
            "notes": rec.get("notes", ""),
        }
        for k, v in rec.get("config", {}).items():
            flat[f"cfg_{k}"] = v
        flat.update(rec.get("metrics", {}))
        rows.append(flat)

    df = pd.DataFrame(rows)

    # Sort by MRR descending (best runs first)
    if "mrr" in df.columns:
        df = df.sort_values("mrr", ascending=False).reset_index(drop=True)

    return df
