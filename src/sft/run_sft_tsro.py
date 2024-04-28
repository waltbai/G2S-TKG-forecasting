import argparse
import json
import logging
import os

from datasets import Dataset
from transformers import AutoTokenizer


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str, default="/data/bailong/models/gpt2"
    )
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--dataset", default="ICEWS14", type=str,
                        choices=["ICEWS14", "ICEWS05-15", "ICEWS18", "WIKI", "YAGO"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_predictions", type=int, default=30)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_evaluate", default=False, action="store_true")
    parser.add_argument("--do_predict", default=True)
    return parser.parse_args()


def train():
    """Train on train and valid set."""


def predict():
    """Predict on test set."""


if __name__ == "__main__":
    logger = logging.getLogger("SFTmodel")
    args = get_args()
    # Load data
    logger.info("Load test data.")
    dataset_path = os.path.join(args.dataset_dir, f"{args.dataset}_TSRO.json")
    with open(dataset_path, "r") as f:
        dataset = Dataset.from_dict(json.load(f))
    train_set = dataset["train"]
    valid_set = dataset["valid"]
    test_set = dataset["test"]
    # Tokenize
    logger.info("Tokenize.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        truncation_side="left"
    )
