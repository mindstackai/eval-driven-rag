"""
Retrieval evaluation metrics for the eval-driven-rag project.
"""
from typing import Any

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Compute recall@k: fraction of relevant docs found in top-k retrieved.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (most relevant first).
        relevant_ids: Ground-truth list of chunk IDs considered relevant.
        k: Cut-off rank.

    Returns:
        Recall@k as a float in [0, 1].  Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Compute MRR: mean of reciprocal ranks of first relevant result.

    For a single query this is simply the reciprocal rank (1/rank) of the
    first retrieved chunk that appears in *relevant_ids*, or 0.0 if none do.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (most relevant first).
        relevant_ids: Ground-truth list of chunk IDs considered relevant.

    Returns:
        Reciprocal rank as a float in [0, 1].
    """
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def chunk_relevance_scores(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
) -> list[float]:
    """
    Compute cosine similarity between query and each retrieved chunk embedding.

    Args:
        query_embedding: Embedding vector for the query.
        chunk_embeddings: List of embedding vectors for each retrieved chunk.

    Returns:
        List of cosine similarity scores (one per chunk), each in [-1, 1].
    """
    q = np.array(query_embedding, dtype=float)
    q_norm = np.linalg.norm(q)

    scores: list[float] = []
    for emb in chunk_embeddings:
        c = np.array(emb, dtype=float)
        c_norm = np.linalg.norm(c)
        if q_norm == 0 or c_norm == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q, c) / (q_norm * c_norm)))
    return scores


def run_retrieval_eval(
    qa_pairs: list[dict],
    retriever: Any,
    k: int = 5,
) -> dict:
    """
    Run full retrieval eval over a list of QA pairs.

    Each QA pair must have at least ``relevant_chunk_ids`` and ``question``
    keys (following the schema in data/eval/qa_pairs.json).

    The *retriever* object must expose a ``get_relevant_documents(query)``
    method that returns LangChain Document objects whose ``metadata`` dict
    contains a ``chunk_id`` key.

    Args:
        qa_pairs: List of QA pair dicts loaded from qa_pairs.json.
        retriever: A LangChain-compatible retriever.
        k: Retrieval cut-off for recall@k.

    Returns:
        Dict with keys:
            - ``recall_at_k`` (float): mean recall@k across all queries.
            - ``mrr`` (float): mean reciprocal rank across all queries.
            - ``avg_chunk_relevance`` (float): mean cosine similarity score
              across all retrieved chunks (requires embeddings to be present
              in Document.metadata["embedding"]; skipped if absent).
    """
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    relevance_scores: list[float] = []

    for pair in qa_pairs:
        question = pair["question"]
        relevant_ids: list[str] = pair["relevant_chunk_ids"]

        docs = retriever.invoke(question)
        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]

        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

        # Cosine relevance (optional — requires embeddings in metadata)
        chunk_embs = [
            d.metadata["embedding"] for d in docs if "embedding" in d.metadata
        ]
        if chunk_embs and "query_embedding" in pair:
            sims = chunk_relevance_scores(pair["query_embedding"], chunk_embs)
            relevance_scores.extend(sims)

    return {
        "recall_at_k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "avg_chunk_relevance": float(np.mean(relevance_scores)) if relevance_scores else None,
    }
