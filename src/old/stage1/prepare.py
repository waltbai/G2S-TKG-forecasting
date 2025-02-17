"""
Llama-factory-like data format.
"""

import json
import logging
import os
import random
import sys

from tqdm import tqdm
from transformers import HfArgumentParser

from src.old.stage1.args import DataArguments
from src.old.stage1.prompt import PromptConstructor
from src.tkg import TKG

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def main():
    """
    Prepare dataset.
    """
    data_args, = HfArgumentParser([DataArguments]).parse_yaml_file(sys.argv[1])

    # Prepare and check paths
    prepare_dir = data_args.prepare_dir
    dataset_dir = data_args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)

    # Construct samples
    prompt_func = PromptConstructor()
    version_suffix = data_args.version_suffix()
    for dataset in data_args.dataset:
        dataset_name = f"{dataset}-{version_suffix}"
        filename = f"{dataset_name}.json"
        data_path = os.path.join(prepare_dir, filename)
        if os.path.exists(data_path) and not data_args.overwrite_cache:
            logger.info(f"Dataset {filename} exists.")
            continue

        # Load dataset
        logger.info(f"Load TKG {dataset}.")
        tkg = TKG.load(dataset_dir, dataset)
        # Construct queries
        logger.info(f"Construct queries.")
        queries = tkg.construct_queries("train")
        if data_args.max_samples is not None:
            random.seed(RANDOM_SEED)
            queries = random.choices(queries, k=data_args.max_samples)
        logger.info(f"Construct {len(queries)} {dataset} queries.")
        # Find history
        logger.info(f"Find history for {dataset} queries.")
        for query in tqdm(queries):
            tkg.find_history(
                query=query,
                strategy=data_args.history_strategy,
                history_length=data_args.history_length,
            )
        # Construct prompts
        logger.info(f"Construct {dataset} prompts.")
        for query in tqdm(queries):
            prompt_func(
                query=query,
                tkg=tkg,
                map_strategy=data_args.map_strategy,
                time_strategy=data_args.time,
                map_entity=data_args.entity,
                map_relation=data_args.relation,
            )

        # Dump file
        data = []
        for query in queries:
            data.append({
                "instruction": "",
                "input": query.prompt,
                "output": query.label,
            })
        with open(data_path, "w") as f:
            json.dump(data, f)

        # Update dataset_info
        dataset_info_path = os.path.join(prepare_dir, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "r") as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        dataset_info.update({
            dataset_name: {
                "file_name": filename,
            }
        })
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f)


if __name__ == "__main__":
    main()
