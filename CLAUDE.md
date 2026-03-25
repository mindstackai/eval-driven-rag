# CLAUDE.md

## Git Workflow
- Always work on a feature branch (e.g. `feat/`, `fix/`, `chore/`)
- Commit when a logical unit of work is complete — not too small (avoid single-line commits), not too large (avoid trunk dumps)
- One commit = one meaningful change (a feature, a fix, a refactor)
- Push branch and open a PR to merge into `main`

## Running the App
- `bash run.sh` to start the Streamlit UI

## Running Evals
- `python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 5`
- `python -m src.eval_retrieval`   # Phase 1: retrieval eval across chunk sizes (no LLM)
- `python -m src.eval_generation`  # Phase 2: generation eval on top configs (uses LLM)

## Running the Benchmark
- `python -m scripts.run_benchmark`               # all 8 models
- `python -m scripts.run_benchmark --skip_finetuned`  # base models only
- `python -m scripts.run_benchmark --model BAAI/bge-small-en-v1.5`  # single model

## Fine-tuning Pipeline
- `python -m scripts.update_training_data`   # Steps 1-3: pairs → negatives → train/eval split
- `python -m src.training.finetune`          # Step 4: fine-tune all 3 HF models

## Environment
- Use `.venv` — activate with `source .venv/bin/activate`
- Copy `.env.example` → `.env` and add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
