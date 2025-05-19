"""
Train a 4-way multilingual intent classifier (Greeting/Farewell/Toxic/Chitchat)
using a compact MiniLM-v2 backbone.

Key features
------------
* ✅  Works with any HF repo that ships `*.bin` **or** `*.safetensors` weights.
* ✅  Resumes interrupted downloads (Git-LFS friendly).
* ✅  CPU-friendly: fits on a 4-core / 8 GB machine.
* ✅  Clean function separation + type hints.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import evaluate
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
)


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #

def load_label_maps(path: str = "config/labels.json") -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read the label ↔︎ id mapping used for training/evaluation."""
    with Path(path).open() as fp:
        label2id: Dict[str, int] = json.load(fp)
    id2label: Dict[int, str] = {v: k for k, v in label2id.items()}
    return label2id, id2label


def resolve_checkpoint(repo_id: str) -> str:
    """Return a *local* path to the first weight file in `repo_id`, downloading if needed."""
    candidates = [f for f in list_repo_files(repo_id) if f.endswith((".bin", ".safetensors"))]
    if not candidates:
        raise RuntimeError(f"No .bin or .safetensors weights found in Hugging Face repo: {repo_id}")

    ckpt_name = candidates[0]
    LOGGER.info("Using checkpoint file: %s", ckpt_name)

    # The file is cached under ~/.cache/huggingface/hub/…
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    return ckpt_path


def encode_example(example, tokenizer, label2id):
    tokens = tokenizer(
        example["text"],
        max_length=48,
        truncation=True,
        padding="max_length",
    )
    tokens["labels"] = label2id[example["label"]]
    return tokens


# --------------------------------------------------------------------------- #
# Main training routine                                                       #
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    label2id, id2label = load_label_maps()

    # 1. Dataset -------------------------------------------------------------- #
    df_train = pd.read_csv(args.train_csv)
    df_valid = pd.read_csv(args.valid_csv)

    ds_train = Dataset.from_pandas(df_train)
    ds_valid = Dataset.from_pandas(df_valid)

    # 2. Tokeniser & encoding ------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id)

    remove_cols = ["text", "label"]  # drop non-tensor columns
    ds_train = ds_train.map(
        lambda ex: encode_example(ex, tokenizer, label2id),
        remove_columns=remove_cols,
    )
    ds_valid = ds_valid.map(
        lambda ex: encode_example(ex, tokenizer, label2id),
        remove_columns=remove_cols,
    )

    # 3. Model ---------------------------------------------------------------- #
    ckpt_path = resolve_checkpoint(args.repo_id)
    config = AutoConfig.from_pretrained(
        args.repo_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, config=config)

    # 4. Training args -------------------------------------------------------- #
    training_args = TrainingArguments(
        output_dir="model",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",  # new param name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=42,
        logging_steps=50,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):  # noqa: D401 — HF Trainer signature
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # 5. Trainer -------------------------------------------------------------- #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        processing_class=tokenizer,  # formerly `tokenizer=...`
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 6. Save ----------------------------------------------------------------- #
    save_dir = Path("model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    LOGGER.info("✅  Model and tokenizer saved to %s", save_dir.resolve())


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM-v2 for intent classification")
    parser.add_argument("--train_csv", default="data/intent_train.csv", help="Path to training CSV")
    parser.add_argument("--valid_csv", default="data/intent_valid.csv", help="Path to validation CSV")
    parser.add_argument("--repo_id", default="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
                        help="HF repo id or local dir of the base encoder")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    main(parser.parse_args())
