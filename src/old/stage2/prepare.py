"""
Llama-factory-like data format.
"""

import json
import logging
import os
import random
import sys

from src.old.stage2.args import get_prepare_args
from src.old.stage2.prompt import PromptConstructor
from src.tkg import TKG

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def main():
    """
    Prepare dataset.
    """
    data_args, = get_prepare_args(sys.argv[1])

    # Prepare and check paths
    prepare_dir = data_args.prepare_dir
    dataset_dir = data_args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)

    # Construct samples
    prompt_func = PromptConstructor()
    version_suffix = data_args.version_suffix()
    for dataset in data_args.dataset:
        logger.info(f"Load TKG {dataset}.")
        tkg = TKG.load(dataset_dir, dataset)
        # Construct queries
        logger.info(f"Construct queries.")
        queries = {}
        for part in ["train", "valid", "test"]:
            queries.setdefault(part, tkg.construct_queries(part))
            if data_args.max_samples is not None and part == "train":
                random.seed(RANDOM_SEED)
                queries["train"] = random.choices(queries["train"], k=data_args.max_samples)
            logger.info(f"Construct {len(queries[part])} {dataset} {part} queries.")
        # Find history
        for part in ["train", "valid", "test"]:
            logger.info(f"Find history for {dataset} {part} queries.")
            for query in queries[part]:
                tkg.find_history(
                    query=query,
                    strategy=data_args.history_strategy,
                    history_length=data_args.history_length,
                )
        # Construct prompts
        for part in ["train", "valid", "test"]:
            logger.info(f"Construct {dataset} {part} prompts.")
            for query in queries[part]:
                prompt_func(
                    query=query,
                    tkg=tkg,
                    map_strategy=data_args.map_strategy,
                    time_strategy=data_args.time,
                    map_entity=data_args.entity,
                    map_relation=data_args.relation,
                )

        # Dump file
        for part in ["train", "valid", "test"]:
            # Write dataset
            dataset_name = f"{dataset}-{part}-{version_suffix}"
            if part == "train":
                dataset_name = f"{dataset_name}-{data_args.max_samples}"
            filename = f"{dataset_name}.json"
            data_path = os.path.join(prepare_dir, filename)
            if os.path.exists(data_path) and not data_args.overwrite_cache:
                logger.info(f"Dataset {filename} exists.")
                continue
            data = []
            for query in queries[part]:
                data.append({
                    "instruction": "",
                    "input": query.prompt,
                    "output": query.label,
                    "filters": query.filters,
                    "id2entity": {v: k for k, v in query.entity_mapping.items()},
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
