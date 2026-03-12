"""Data loading and schema validation for eval result JSON files."""
import json
import logging
import os
from datetime import datetime
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)

# Fields required in every result file
_COMMON_REQUIRED = {"chunk_size", "chunking_strategy", "timestamp"}
_RETRIEVAL_REQUIRED = _COMMON_REQUIRED | {"hit_rate", "mrr"}
_GENERATION_REQUIRED = _COMMON_REQUIRED | {"avg_correctness", "avg_faithfulness", "avg_relevance"}


def format_timestamp(ts):
    """Format raw timestamp string (e.g. '20260310_162357') for display."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ts


def make_display_name(data):
    """Build a display_name from result data, with fallback for old runs.

    Priority:
    1. Explicit display_name in data
    2. run_name + formatted timestamp
    3. Auto-label from strategy_chunksizes_timestamp
    4. Formatted timestamp alone
    """
    if data.get("display_name"):
        return data["display_name"]
    if data.get("run_name"):
        ts_fmt = format_timestamp(data.get("timestamp", ""))
        return f"{data['run_name']} ({ts_fmt})"
    # Auto-label from config params
    strategy = data.get("chunking_strategy", "")
    chunk_size = data.get("chunk_size", "")
    ts = data.get("timestamp", "")
    ts_fmt = format_timestamp(ts)
    if strategy and chunk_size:
        return f"{strategy}_{chunk_size} ({ts_fmt})"
    return ts_fmt or ts or "unknown"


def _validate_fields(data, required, filepath):
    """Warn if required fields are missing from a result file."""
    missing = required - set(data.keys())
    if missing:
        logger.warning("Missing fields %s in %s", missing, filepath)
    return len(missing) == 0


def load_all_results(results_dir):
    """Load all retrieval and generation result files into DataFrames.

    Args:
        results_dir: Path to directory containing eval result JSON files.

    Returns:
        Tuple of (retrieval_df, generation_df). Either may be empty.
    """
    retrieval_rows = []
    generation_rows = []

    pattern = os.path.join(results_dir, "*.json")
    files = sorted(glob(pattern))
    logger.info("Found %d JSON files in %s", len(files), results_dir)

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", fpath, e)
            continue

        display_name = make_display_name(data)

        if fname.startswith("retrieval_"):
            _validate_fields(data, _RETRIEVAL_REQUIRED, fpath)
            retrieval_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "run_name": data.get("run_name", ""),
                "display_name": display_name,
                "hit_rate": data.get("hit_rate", 0),
                "mrr": data.get("mrr", 0),
                "recall_at_k": data.get("recall_at_k", 0),
                "avg_retrieval_confidence": data.get("avg_retrieval_confidence", 0),
                "per_question": data.get("per_question", []),
            })
            logger.info("Loaded retrieval: %s (strategy=%s, chunk_size=%s)",
                        fname, data.get("chunking_strategy"), data.get("chunk_size"))

        elif fname.startswith("generation_"):
            _validate_fields(data, _GENERATION_REQUIRED, fpath)
            generation_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "run_name": data.get("run_name", ""),
                "display_name": display_name,
                "avg_correctness": data.get("avg_correctness", 0),
                "avg_faithfulness": data.get("avg_faithfulness", 0),
                "avg_relevance": data.get("avg_relevance", 0),
                "cache_hits": data.get("cache_hits", 0),
                "phase1_hit_rate": data.get("phase1_hit_rate", 0),
                "phase1_mrr": data.get("phase1_mrr", 0),
                "avg_retrieval_confidence": data.get("avg_retrieval_confidence", 0),
                "per_question": data.get("per_question", []),
            })
            logger.info("Loaded generation: %s (strategy=%s, chunk_size=%s)",
                        fname, data.get("chunking_strategy"), data.get("chunk_size"))

    ret_df = pd.DataFrame(retrieval_rows) if retrieval_rows else pd.DataFrame()
    gen_df = pd.DataFrame(generation_rows) if generation_rows else pd.DataFrame()

    logger.info("Total: %d retrieval rows, %d generation rows", len(ret_df), len(gen_df))
    return ret_df, gen_df
