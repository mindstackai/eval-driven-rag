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

## Environment
- Use `.venv` — activate with `source .venv/bin/activate`
- Copy `.env.example` → `.env` and add `OPENAI_API_KEY`
