# eval-driven-rag
### MindStack AI · github.com/mindstackai/eval-driven-rag

---

## What This Is

A production-grade RAG system built around a single principle: **you should know if your retrieval is working before it hits a user.**

Most RAG tutorials stop at "it returns an answer." This repo goes further — it builds the eval layer in from the start, treating retrieval quality and answer faithfulness as first-class concerns, not afterthoughts.

Built and maintained by [Olivia Chen](https://mindstackai.substack.com) as part of MindStack AI.

> 📬 **Read the deep-dive:** [How I think about evals for RAG pipelines →](https://mindstackai.substack.com) *(coming soon)*

---

## Why eval-driven?

RAG systems fail in quiet, hard-to-catch ways:

- The right document is in your index — but it's not being retrieved
- The retrieved chunks are close but not grounded enough for the LLM to answer correctly
- Answer quality degrades on edge cases that never showed up in your happy-path testing

This repo treats **evals as infrastructure**, not a nice-to-have. Every retrieval decision has a measurable signal.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         eval-driven-rag         │
                        └─────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
  │   Documents  │────▶│   Ingestion  │────▶│    Vector Store      │
  │  (PDF, text) │     │              │     │  FAISS  │  LanceDB   │
  └──────────────┘     │ • chunking   │     └────────────────────-─┘
                       │ • embedding  │               │
                       │ • metadata   │               │ retrieve top-k
                       └──────────────┘               ▼
                                              ┌──────────────────┐
  ┌──────────────┐                            │    Retriever     │
  │  User Query  │ ──────────────────────────▶│                  │
  └──────────────┘                            │ • dense search   │
                                              │ • metadata filter│
                                              └────────┬─────────┘
                                                       │
                                          ┌────────────▼──────────┐
                                          │     Eval Layer         ◀────── eval-driven
                                          │                        │
                                          │ • recall@k             │
                                          │ • MRR                  │
                                          │ • faithfulness score   │
                                          │ • answer relevance     │
                                          └────────────┬───────────┘
                                                       │
                                          ┌────────────▼───────────┐
                                          │     LLM Generation     │
                                          │  (grounded, cited)     │
                                          └────────────────────────┘
                                                       │
                                          ┌────────────▼───────────┐
                                          │    Streamlit UI        │
                                          └────────────────────────┘
```

---

## Repository Structure

```
eval-driven-rag/
│
├── data/
│   ├── raw/                        # source documents (PDF, txt)
│   ├── processed/                  # preprocessed text
│   └── eval/
│       └── qa_pairs.json           # labeled Q&A pairs for eval runs
│
├── notebooks/
│   ├── 01_chunk_token_vector.ipynb # chunking strategy exploration
│   ├── 02_retrieval_eval.ipynb     # eval framework walkthrough
│   └── 03_lancedb_vs_faiss.ipynb   # vector store comparison
│
├── src/
│   ├── ingest.py                   # load → split → embed → store
│   ├── splitters.py                # chunking strategies
│   ├── embed_store.py              # vector store build/load
│   ├── retriever.py                # retriever abstraction
│   ├── app.py                      # Streamlit RAG UI
│   │
│   ├── vectorstore/
│   │   ├── faiss_store.py          # FAISS implementation
│   │   └── lancedb_store.py        # LanceDB implementation (NEW)
│   │
│   └── eval/
│       ├── __init__.py
│       ├── retrieval_eval.py       # recall@k, MRR, chunk relevance
│       ├── answer_eval.py          # faithfulness, answer relevance
│       └── run_eval.py             # CLI entrypoint for eval runs
│
├── docs/
│   └── design_decisions.md         # why we made the choices we made
│
├── .env.example
├── config.yaml
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Eval Framework Design

### Retrieval Eval (`src/eval/retrieval_eval.py`)

**recall@k**
> Of the relevant documents for a query, how many appear in the top-k retrieved chunks?

```python
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float
```

**MRR (Mean Reciprocal Rank)**
> Where does the first relevant chunk appear in the ranked results?

```python
def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float
```

**Chunk Relevance Score**
> Cosine similarity between query embedding and retrieved chunk embeddings.

```python
def chunk_relevance_scores(query_embedding, chunk_embeddings) -> list[float]
```

---

### Answer Eval (`src/eval/answer_eval.py`)

**Faithfulness**
> Is the generated answer grounded in the retrieved context, or is the LLM hallucinating?

```python
def faithfulness_score(answer: str, context_chunks: list[str]) -> float
```

**Answer Relevance**
> Does the answer actually address the user's question?

```python
def answer_relevance_score(question: str, answer: str) -> float
```

---

### Eval Test Set (`data/eval/qa_pairs.json`)

A small labeled dataset — even 15-20 pairs is enough to catch regressions.

```json
[
  {
    "question": "What is the maintenance schedule for vehicle class A?",
    "relevant_chunk_ids": ["doc_001_chunk_3", "doc_001_chunk_4"],
    "reference_answer": "Vehicle class A requires quarterly maintenance..."
  }
]
```

---

## Vector Store: FAISS vs LanceDB

| | FAISS | LanceDB |
|---|---|---|
| **Setup** | In-memory, local file | Embedded DB, persistent |
| **Metadata filtering** | Limited | Native, expressive |
| **Scale** | Good for prototyping | Production-ready |
| **Access control** *(future)* | Not supported | Supported via metadata |
| **Use this when** | Fast local dev | Production + filtering |

> ⚠️ **Planned:** File-level access control via metadata filtering in LanceDB.
> After embedding, permissions will be stored in a DB and applied at query time —
> so users only retrieve chunks from documents they're authorized to see.
> *[Research in progress — contributions welcome]*

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/mindstackai/eval-driven-rag.git
cd eval-driven-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your OPENAI_API_KEY

# 3. Add documents
# Drop PDFs or .txt files into data/raw/

# 4. Ingest + embed
python -m src.ingest

# 5. Run the app
bash run.sh
# Or directly:
# KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/app.py

# 6. Run evals
python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 5
```

---

## Design Decisions

See [`docs/design_decisions.md`](docs/design_decisions.md) for the full reasoning behind:
- Chunking strategy choices
- Why we support both FAISS and LanceDB
- How the eval test set was constructed
- Tradeoffs in the faithfulness scoring approach

---

## Roadmap

- [x] End-to-end RAG pipeline (ingest → retrieve → generate)
- [x] FAISS vector store
- [x] Streamlit UI
- [ ] Eval framework (recall@k, MRR, faithfulness)
- [ ] LanceDB vector store
- [ ] BM25 + dense hybrid search
- [ ] Cross-encoder reranker
- [ ] File-level access control via metadata + permissions DB
- [ ] Companion Substack post

---

## About

Part of [MindStack AI](https://mindstackai.substack.com) — writing and code on eval-driven AI design,
ML in production, and data architecture for AI systems.

---

*The systems we build matter. Let's build them well.*
