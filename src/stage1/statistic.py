import json
import logging
import os
from typing import Dict, Any

from transformers import AutoTokenizer, HfArgumentParser

from llamafactory.model import load_tokenizer
from src.stage1.prepare import get_data_version
from src.utils.args import ModelArguments, AnonymizedDataArguments

logger = logging.getLogger(__name__)


def dataset_statistics(
        dataset: Dict[str, Any],
        model_args: ModelArguments,
):
    """Dataset basic statistics."""
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    for part in ["train", "valid", "test"]:
        ds = dataset[part]
        logger.info(f"Statistics on {part} set.")

        # Prompt length statistic
        len_prompts = [
            len(tokenizer(prompt)["input_ids"]) for prompt in ds["prompt"]
        ]
        avg_len = sum(len_prompts) / len(len_prompts)
        num_longer_than_1024 = len([_ for _ in len_prompts if _ > 1024])
        num_longer_than_2048 = len([_ for _ in len_prompts if _ > 2048])
        logger.info(f"Average token length: {avg_len:.2f}.")
        logger.info(f"Ratio of prompts longer than 1024: "
                    f"{num_longer_than_1024}/{len(len_prompts)}.")
        logger.info(f"Ratio of prompts longer than 2048: "
                    f"{num_longer_than_2048}/{len(len_prompts)}.")

        # Out-of-history statistic
        out_of_history = 0
        for label, candidates in zip(ds["label"], ds["candidates"]):
            if label not in candidates:
                out_of_history += 1
        logger.info(f"Ratio of out-of-history queries: "
                    f"{out_of_history}/{len(ds['label'])}. ")


if __name__ == "__main__":
    # Parse arguments from config file
    parser = HfArgumentParser([AnonymizedDataArguments, ModelArguments])
    data_args, model_args = parser.parse_args_into_dataclasses()
    # Prepare
    datafile_name = get_data_version(data_args) + ".json"
    data_path = os.path.join(data_args.prepare_dir, datafile_name)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    dataset_statistics(dataset, model_args)
