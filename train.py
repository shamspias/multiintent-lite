"""
Fine-tune MiniLM-v2 on 4-way intent classification (multilingual).
"""
import evaluate
import json, argparse, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)


def load_label_maps(path="config/labels.json"):
    with open(path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def main(args):
    label2id, id2label = load_label_maps()

    # 1️⃣  Dataset ----------------------------------------------------------------
    df_train = pd.read_csv(args.train_csv)
    df_valid = pd.read_csv(args.valid_csv)
    ds_train = Dataset.from_pandas(df_train)
    ds_valid = Dataset.from_pandas(df_valid)

    # 2️⃣  Tokeniser --------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "nreimers/mmarco-mMiniLMv2-L6-H384-v1"
    )

    def encode(batch):
        enc = tokenizer(
            batch["text"],
            max_length=48,
            padding="max_length",
            truncation=True
        )
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    ds_train = ds_train.map(encode, batched=True, remove_columns=["text", "label"])
    ds_valid = ds_valid.map(encode, batched=True, remove_columns=["text", "label"])

    # 3️⃣  Model ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 4️⃣  Training --------------------------------------------------------------
    args_out = TrainingArguments(
        output_dir="model",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=42,
        logging_steps=50,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("model")
    tokenizer.save_pretrained("model")
    print("✅  Model saved to ./model")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/intent_train.csv")
    p.add_argument("--valid_csv", default="data/intent_valid.csv")
    main(p.parse_args())
