"""
Fine-tune MiniLM-v2 on a 4-class multilingual intent dataset
(Greeting / Farewell / Toxic / Chitchat).

Usage
-----
python train.py \
    --train_csv data/intent_train.csv \
    --valid_csv data/intent_valid.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import evaluate
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class FineTuneConfig:
    repo_id: str = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    max_len: int = 48
    weight_decay: float = 1e-2
    output_dir: str = "model"
    seed: int = 42
    label_path: str = "config/labels.json"


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #

def load_label_maps(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with Path(path).open() as fp:
        label2id: Dict[str, int] = json.load(fp)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def resolve_checkpoint(repo_id: str) -> str:
    """Return the first *.bin / *.safetensors file from `repo_id`, downloading if needed."""
    ckpts: List[str] = [
        f for f in list_repo_files(repo_id) if f.endswith((".bin", ".safetensors"))
    ]
    if not ckpts:
        raise RuntimeError(f"No weight files found in repo {repo_id}")
    return hf_hub_download(repo_id, filename=ckpts[0])


# --------------------------------------------------------------------------- #
# Main trainer class                                                          #
# --------------------------------------------------------------------------- #

class MultiIntentFineTuner:
    def __init__(self, cfg: FineTuneConfig, train_csv: str, valid_csv: str) -> None:
        self.cfg = cfg
        self.train_csv = train_csv
        self.valid_csv = valid_csv

        # logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        )
        self.log = logging.getLogger(self.__class__.__name__)

        # labels
        self.label2id, self.id2label = load_label_maps(cfg.label_path)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.repo_id)

        # device-specific pin-memory: safe for CUDA, off for Apple MPS
        self.pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()

    # --------------------------------------------------------------------- #
    # Data preparation                                                      #
    # --------------------------------------------------------------------- #

    def _encode(self, example):
        toks = self.tokenizer(
            example["text"],
            max_length=self.cfg.max_len,
            truncation=True,
            padding="max_length",
        )
        toks["labels"] = self.label2id[example["label"]]
        return toks

    def load_datasets(self):
        df_train = pd.read_csv(self.train_csv)
        df_valid = pd.read_csv(self.valid_csv)

        ds_train = Dataset.from_pandas(df_train)
        ds_valid = Dataset.from_pandas(df_valid)

        remove = ["text", "label"]  # drop raw string columns
        self.ds_train = ds_train.map(self._encode, remove_columns=remove)
        self.ds_valid = ds_valid.map(self._encode, remove_columns=remove)

    # --------------------------------------------------------------------- #
    # Model, arguments, trainer                                             #
    # --------------------------------------------------------------------- #

    def build_model(self):
        ckpt = resolve_checkpoint(self.cfg.repo_id)
        cfg = AutoConfig.from_pretrained(
            self.cfg.repo_id,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            config=cfg,
            ignore_mismatched_sizes=True,  # new head -> avoid noisy INFO line
        )

    def build_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            seed=self.cfg.seed,
            logging_steps=50,
            dataloader_pin_memory=self.pin_memory,
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):  # noqa: D401
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            return metric.compute(predictions=preds, references=labels)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds_train,
            eval_dataset=self.ds_valid,
            processing_class=self.tokenizer,  # supersedes `tokenizer`
            compute_metrics=compute_metrics,
        )

    # --------------------------------------------------------------------- #
    # Orchestration                                                         #
    # --------------------------------------------------------------------- #

    def run(self):
        self.log.info("Configuration: %s", asdict(self.cfg))
        self.load_datasets()
        self.build_model()
        self.build_trainer()
        self.trainer.train()
        self.save()

    def save(self):
        save_dir = Path(self.cfg.output_dir)
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.log.info("✅  Model and tokenizer saved to %s", save_dir.resolve())


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM-v2 for intent classification")
    parser.add_argument("--train_csv", default="data/intent_train.csv")
    parser.add_argument("--valid_csv", default="data/intent_valid.csv")
    parser.add_argument("--repo_id", default=FineTuneConfig.repo_id)
    parser.add_argument("--epochs", type=int, default=FineTuneConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument("--lr", type=float, default=FineTuneConfig.lr)
    args = parser.parse_args()

    cfg = FineTuneConfig(
        repo_id=args.repo_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    trainer = MultiIntentFineTuner(cfg, args.train_csv, args.valid_csv)
    trainer.run()
