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
│       ├── qa_pairs.json           # labeled Q&A pairs for eval runs
│       └── eval_queries.jsonl      # benchmark eval queries (0-based corpus positions)
│
├── notebooks/
│   ├── 01_chunk_token_vector.ipynb # chunking strategy exploration
│   ├── 02_retrieval_eval.ipynb     # eval framework walkthrough
│   ├── 03_lancedb_vs_faiss.ipynb   # vector store comparison
│   └── finetune_colab.ipynb        # fine-tuning notebook (Colab / GPU)
│
├── eval/
│   ├── ground_truth.json           # labeled Q&A pairs for two-phase eval
│   ├── cache/                      # cached LLM responses (Phase 2)
│   └── results/                    # eval results (JSON per run)
│
├── scripts/
│   ├── run_benchmark.py            # 8-model embedding benchmark (Hit@5, MRR, Recall@5)
│   ├── smoke_test_embedders.py     # verify all embedders load and produce correct dims
│   └── update_training_data.py     # one-command Phase 2 data refresh (Steps 1-3)
│
├── traces/
│   └── experiment_log.jsonl        # benchmark run history (all models, all runs)
│
├── src/
│   ├── ingest.py                   # load → split → embed → store
│   ├── splitters.py                # chunking strategies
│   ├── embed_store.py              # vector store build/load
│   ├── retriever.py                # retriever abstraction
│   ├── tracing.py                  # EvalTrace span instrumentation
│   ├── app.py                      # Streamlit RAG UI
│   ├── eval_retrieval.py           # Phase 1: retrieval eval (no LLM)
│   ├── eval_generation.py          # Phase 2: generation eval (LLM)
│   ├── eval_dashboard.py           # legacy entry point (redirects to dashboard module)
│   │
│   ├── embedders/                  # embedding model abstraction
│   │   ├── __init__.py             # load_embedder() factory
│   │   ├── base.py                 # BaseEmbedder ABC
│   │   ├── huggingface.py          # HuggingFaceEmbedder (local, MPS/CUDA/CPU)
│   │   └── openai.py               # OpenAIEmbedder (API)
│   │
│   ├── training/                   # fine-tuning pipeline
│   │   ├── generate_pairs.py       # synthetic Q&A via Claude API (idempotent)
│   │   ├── mine_negatives.py       # hard negative mining via LanceDB (3 tiers)
│   │   ├── false_negative_check.py # flag false negatives + train/eval split
│   │   └── finetune.py             # SentenceTransformerTrainer + MNRL loss
│   │
│   ├── dashboard/                  # modular eval dashboard
│   │   ├── __init__.py             # exports render_dashboard(results_dir, config_path)
│   │   ├── app.py                  # standalone entry point
│   │   ├── components/
│   │   │   ├── sidebar.py          # filter sidebar (strategy, run, select/clear all)
│   │   │   ├── phase1.py           # Phase 1 retrieval charts + metrics
│   │   │   ├── phase2.py           # Phase 2 generation charts + metrics
│   │   │   ├── comparison.py       # config comparison tables + strategy comparison
│   │   │   ├── model_comparison.py # leaderboard, MRR bar chart, quality-vs-cost scatter
│   │   │   └── run_eval.py         # run Phase 1/Phase 2 eval forms
│   │   └── utils/
│   │       ├── data_loader.py      # JSON loading + schema validation
│   │       └── metrics.py          # chart helpers + score calculations
│   │
│   ├── vectorstore/
│   │   ├── faiss_store.py          # FAISS implementation
│   │   └── lancedb_store.py        # LanceDB implementation
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
    "question": "What encoding method is used to map classical data into the quantum circuit?",
    "relevant_chunk_ids": ["data/raw/2603.06473v1.pdf:p2:c19:3a901645"],
    "reference_answer": "Angle encoding is used, where rotations on the Y-axis of qubits encode the features."
  }
]
```

---

## Two-Phase Eval System

A systematic approach to finding the best chunking config for your documents.

### Phase 1: Retrieval Eval (no LLM, local only)

Tests multiple chunk sizes, re-ingests docs for each, and ranks by retrieval quality.

```bash
# Run with defaults from config.yaml
python -m src.eval_retrieval

# Name your run for easy identification in the dashboard
python -m src.eval_retrieval --name "baseline recursive strategy"

# Custom ground truth file
python -m src.eval_retrieval --ground_truth eval/ground_truth.json

# Save output to file while still printing to screen
python -m src.eval_retrieval 2>&1 | tee eval/results/phase1_log.txt
```

**What it does:**
1. For each chunk size in `eval.chunk_sizes_to_test` (default: [128, 256, 512, 1024]):
   - Re-chunks all documents with that size
   - Builds a temporary FAISS index
   - Runs retrieval for each ground truth question
   - Uses content-based matching (not exact chunk ID) so it works across chunk sizes
2. Calculates Hit Rate, MRR, and Recall@k per config
3. Saves per-config results to `eval/results/retrieval_{chunk_size}_{timestamp}.json`
4. Prints a ranked summary table

**Example output:**
```
========================================================================
 Phase 1: Retrieval Eval Summary (ranked by Hit Rate)
========================================================================
  Rank   Chunk Size   Chunks   Hit Rate   MRR        Recall@4
------------------------------------------------------------------------
  1      512          227      0.8750     0.6979     0.7500
  2      128          1323     0.7500     0.6250     0.6875
  3      256          602      0.7500     0.5938     0.6250
  4      1024         109      0.7500     0.6875     0.6250
========================================================================
```

### Phase 2: Generation Eval (LLM, top configs only)

Takes the best chunk configs from Phase 1 and evaluates full RAG generation quality.

```bash
# Run with defaults (top 4 configs from Phase 1)
python -m src.eval_generation

# Name your run
python -m src.eval_generation --name "testing semantic generation"

# Evaluate top 3 configs instead
python -m src.eval_generation --top_n 3

# Save output to file
python -m src.eval_generation 2>&1 | tee eval/results/phase2_log.txt
```

**What it does:**
1. Reads Phase 1 results and takes top N configs by hit rate
2. For each config: rebuilds index, runs full RAG with LLM for each question
3. Caches LLM responses in `eval/cache/` (key = hash of question + retrieved chunks)
4. Scores each answer on correctness, faithfulness, and relevance
5. Saves results to `eval/results/generation_{chunk_size}_{timestamp}.json`

**Example output:**
```
==========================================================================================
 Phase 2: Generation Eval Summary (ranked by Correctness)
==========================================================================================
  Rank   Chunk    Chunks   Correct    Faithful   Relevant   HitRate    MRR
------------------------------------------------------------------------------------------
  1      512      227      0.7500     1.0000     0.8375     0.8750     0.6979
  2      256      602      0.6125     0.8750     0.8125     0.7500     0.5938
==========================================================================================
```

### Ground Truth Format (`eval/ground_truth.json`)

```json
[
  {
    "question": "Your question here",
    "expected_chunk_ids": ["source:pPage:cIndex:hash"],
    "expected_answer": "The expected answer for correctness scoring."
  }
]
```

### Config (`config.yaml`)

```yaml
eval:
  chunk_sizes_to_test: [128, 256, 512, 1024]
  top_k_configs_for_generation: 4
  cache_responses: true
  cache_dir: eval/cache/
  results_dir: eval/results/
```

---

## Eval Dashboard

A Streamlit dashboard for visualizing eval results across chunk configs and strategies.

```bash
# Standalone
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/dashboard/app.py

# Or via the helper script
bash run_dashboard.sh

# Legacy entry point (still works)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/eval_dashboard.py
```

**Dashboard features:**
- **Phase 1 & Phase 2 sections** — grouped bar charts with data labels, colored by strategy
- **Strategy Comparison** — side-by-side metrics when multiple strategies are selected
- **Config Comparison tables** — overall score, recommended badge, grouped by strategy
- **Sidebar filters** — filter by strategy and eval run (uses run names, not raw timestamps)
- **Run eval from UI** — Phase 1 and Phase 2 forms with config preview
- **Importable** — use in your own Streamlit app:

```python
from src.dashboard import render_dashboard
render_dashboard(results_dir="eval/results", config_path="config.yaml")
```

---

## Embedding Model Benchmark

A systematic comparison of 8 embedding models — 3 open-source HuggingFace models (base + fine-tuned) and 2 OpenAI API models — evaluated at `chunk_size=512` on Hit@5, MRR, and Recall@5.

### Results (on technical PDF corpus)

| Rank | Model | Source | Fine-tuned | Hit@5 | MRR | Recall@5 | Cost/1k |
|---|---|---|---|---|---|---|---|
| 🥇 1 | bge-small-en-v1.5-finetuned | local | ✓ | 0.9500 | **0.8211** | 0.9500 | $0 |
| 2 | bge-base-en-v1.5-finetuned | local | ✓ | 0.9500 | 0.8111 | 0.9500 | $0 |
| 3 | BAAI/bge-base-en-v1.5 | local | | 0.9167 | 0.8006 | 0.9167 | $0 |
| 4 | all-MiniLM-L6-v2-finetuned | local | ✓ | 0.9333 | 0.7950 | 0.9333 | $0 |
| 5 | text-embedding-3-small | openai | | 0.9333 | 0.7831 | 0.9333 | $0.00002 |
| 6 | text-embedding-3-large | openai | | 0.9500 | 0.7761 | 0.9500 | $0.00013 |
| 7 | BAAI/bge-small-en-v1.5 | local | | 0.9000 | 0.7719 | 0.9000 | $0 |
| 8 | all-MiniLM-L6-v2 | local | | 0.8500 | 0.7289 | 0.8500 | $0 |

> **Key finding:** Fine-tuned local models beat OpenAI API models on MRR (0.82 vs 0.78) at $0 inference cost.

### How to run the benchmark

```bash
# Run all 8 models
python -m scripts.run_benchmark

# Skip fine-tuned models (base models only, no checkpoints needed)
python -m scripts.run_benchmark --skip_finetuned

# Single model
python -m scripts.run_benchmark --model BAAI/bge-small-en-v1.5
```

Results are logged to `traces/experiment_log.jsonl` and visible in the **Model Comparison** tab of the dashboard.

### Fine-tuning pipeline

Fine-tuning uses domain-specific synthetic Q&A pairs generated from your corpus, with hard negative mining via LanceDB.

```bash
# Step 1 — generate synthetic Q&A pairs (Claude API, idempotent)
# Step 2 — mine hard negatives (LanceDB, 3 tiers: very_hard/hard/medium)
# Step 3 — flag false negatives + train/eval split
python -m scripts.update_training_data

# Step 4 — fine-tune all 3 HF models (GPU recommended — use Colab for M2 Macs)
python -m src.training.finetune
```

Fine-tuned checkpoints are saved to `finetuned/{model-slug}-finetuned/`. Add `ANTHROPIC_API_KEY` to `.env` for Step 1.

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

# 7. Two-phase eval (find optimal chunk size)
python -m src.eval_retrieval --name "baseline run"    # Phase 1: retrieval only (no LLM cost)
python -m src.eval_generation --name "baseline run"   # Phase 2: generation quality (uses LLM)

# 8. Run embedding model benchmark (compare 8 models)
python -m scripts.run_benchmark --skip_finetuned      # base models only
python -m scripts.run_benchmark                       # all 8 including fine-tuned

# 9. View results in the dashboard
bash run_dashboard.sh
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
- [x] LanceDB vector store
- [x] Streamlit UI
- [x] Eval framework (recall@k, MRR, faithfulness)
- [x] Two-phase eval system (retrieval + generation across chunk sizes)
- [x] LLM response caching for eval runs
- [x] EvalTrace integration (span tracing, latency, cost, SLO)
- [x] Eval dashboard with strategy comparison and run naming
- [x] BaseEmbedder abstraction (HuggingFace + OpenAI)
- [x] Fine-tuning pipeline (synthetic pairs → hard negatives → MNRL training)
- [x] 8-model embedding benchmark with Model Comparison dashboard tab
- [x] Colab fine-tuning notebook (M2 Mac OOM workaround)
- [ ] BM25 + dense hybrid search
- [ ] Cross-encoder reranker
- [ ] Chunk size × embedding model grid search
- [ ] File-level access control via metadata + permissions DB
- [ ] Companion Substack post

---

## About

Part of [MindStack AI](https://mindstackai.substack.com) — writing and code on eval-driven AI design,
ML in production, and data architecture for AI systems.

---

*The systems we build matter. Let's build them well.*
