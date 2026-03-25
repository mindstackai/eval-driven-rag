"""
Fine-tune a SentenceTransformer model on cleaned triplets using
MultipleNegativesRankingLoss (MNRL).

MNRL treats all other positives in the batch as implicit negatives, so
larger batch sizes = more in-batch negatives = better training signal.

M2 Mac constraints:
    - fp16=False, bf16=False  (MPS does not support mixed precision)
    - If OOM: reduce batch_size to 8 in config.yaml
    - If MPS op error: set PYTORCH_ENABLE_MPS_FALLBACK=1 in environment
    - Expect ~2-4 hours per model on M2 for 500 pairs × 3 epochs

Usage:
    from src.training.finetune import finetune_model
    output_path = finetune_model("BAAI/bge-small-en-v1.5", config)

    # Or run all three models from CLI:
    python -m src.training.finetune
"""

import json
import re
from pathlib import Path

FINETUNED_DIR = Path("finetuned")
TRAIN_PATH = Path("data/train.jsonl")
EVAL_PATH = Path("data/eval.jsonl")


def _slug(model_name: str) -> str:
    """BAAI/bge-small-en-v1.5 → bge-small-en-v1-5"""
    name = model_name.split("/")[-1]
    return re.sub(r"[^a-z0-9-]", "-", name.lower())


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_dataset(triplets: list[dict]):
    """Convert triplets to a HuggingFace Dataset with anchor/positive columns."""
    from datasets import Dataset

    records = [{"anchor": t["query"], "positive": t["positive"]} for t in triplets]
    return Dataset.from_list(records)


def finetune_model(
    base_model_name: str,
    config: dict,
    train_path: Path = TRAIN_PATH,
    eval_path: Path = EVAL_PATH,
) -> str:
    """
    Fine-tune base_model_name on train_path, evaluate on eval_path.

    Args:
        base_model_name: HuggingFace model ID (e.g. "BAAI/bge-small-en-v1.5").
        config: Parsed config.yaml dict.
        train_path: Path to data/train.jsonl.
        eval_path: Path to data/eval.jsonl.

    Returns:
        Output path string where the fine-tuned model is saved.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    training_cfg = config.get("training", {})
    epochs = training_cfg.get("epochs", 3)
    batch_size = training_cfg.get("batch_size", 16)
    lr = training_cfg.get("learning_rate", 2e-5)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.1)

    output_path = str(FINETUNED_DIR / f"{_slug(base_model_name)}-finetuned")
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {base_model_name}")
    print(f"Output:      {output_path}")
    print(f"Epochs:      {epochs}  |  Batch: {batch_size}  |  LR: {lr}")

    # Load model — device auto-detected by HuggingFaceEmbedder logic
    model = SentenceTransformer(base_model_name)

    # Load datasets
    train_triplets = _load_jsonl(train_path)
    eval_triplets = _load_jsonl(eval_path)
    print(f"Train pairs: {len(train_triplets)}  |  Eval pairs: {len(eval_triplets)}")

    train_dataset = _build_dataset(train_triplets)
    eval_dataset = _build_dataset(eval_triplets)

    # MNRL: anchor + positive only — other positives in batch are implicit negatives
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=int(warmup_ratio * len(train_dataset) / batch_size * epochs),
        # NOTE: MPS does not support fp16/bf16 — keep both False
        fp16=False,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    return output_path


def main():
    """Fine-tune all three local HF models listed in config.yaml."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    local_models = [
        m["name"]
        for m in config["embedding"]["models"]
        if m["source"] == "local"
    ]

    if not TRAIN_PATH.exists() or not EVAL_PATH.exists():
        print("ERROR: data/train.jsonl or data/eval.jsonl not found.")
        print("Run Phase 2 first: generate_pairs → mine_negatives → false_negative_check → split_train_eval")
        return

    results = {}
    for model_name in local_models:
        output_path = finetune_model(model_name, config)
        results[model_name] = output_path

    print(f"\n{'='*60}")
    print("Fine-tuning complete:")
    for name, path in results.items():
        print(f"  {name} → {path}")


if __name__ == "__main__":
    main()
