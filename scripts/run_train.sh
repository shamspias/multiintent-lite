#!/usr/bin/env bash
python train.py --train_csv data/intent_train.csv --valid_csv data/intent_valid.csv
### With huggingface repo
# python train.py --train_csv data/intent_train.csv --valid_csv data/intent_valid.csv --repo_id prajjwal1/bert-tiny