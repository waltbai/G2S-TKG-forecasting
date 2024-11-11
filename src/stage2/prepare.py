import json
import logging
import os
import sys

from src.stage2.args import get_prepare_args, DataArguments
from src.stage2.prompt import PromptConstructor
from src.utils.data.tkg import TKG

logger = logging.getLogger(__name__)


def prepare(data_args: DataArguments):
    """
    Prepare dataset.
    """
    # Prepare and check paths
    prepare_dir = data_args.prepare_dir
    dataset_dir = data_args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)
    datafile_name = data_args.get_data_version() + ".json"
    data_path = os.path.join(prepare_dir, datafile_name)
    if os.path.exists(data_path) and not data_args.overwrite_cache:
        logger.info(f"Overwrite existing dataset.")
        return data_path
    else:
        logger.info(f"Prepare dataset.")

    # Construct samples
    queries = {"train": [], "valid": [], "test": []}
    prompt_func = PromptConstructor()
    for dataset in data_args.dataset:
        logger.info(f"Load TKG {dataset}.")
        tkg = TKG.load(dataset_dir, dataset)
        # Construct queries
        logger.info(f"Construct queries.")
        tmp_queries = {}
        for part in ["train", "valid", "test"]:
            tmp_queries.setdefault(part, tkg.construct_queries(part))
            logger.info(f"Construct {len(tmp_queries[part])} {dataset} {part} queries.")
        # Find history
        for part in ["train", "valid", "test"]:
            logger.info(f"Find history for {dataset} {part} queries.")
            for query in tmp_queries[part]:
                tkg.find_history(query, data_args.history_strategy)
        # Construct prompts
        for part in ["train", "valid", "test"]:
            logger.info(f"Construct {dataset} {part} prompts.")
            for query in tmp_queries[part]:
                tkg.find_history(query, data_args.history_strategy)
                prompt_func(
                    query=query,
                    tkg=tkg,
                    map_entity=data_args.entity,
                    map_relation=data_args.relation,
                )
        for part in ["train", "valid", "test"]:
            queries[part].extend(tmp_queries[part])

    data = {}
    for part in ["train", "valid", "test"]:
        data.setdefault(part, {})
        for query in queries[part]:
            data[part].setdefault("prompt", []).append(query.prompt)
            data[part].setdefault("label", []).append(query.label)
            data[part].setdefault("filters", []).append(query.filters)
            data[part].setdefault("id2entity", []).append(
                {v: k for k, v in query.entity_mapping.items()}
            )
    with open(data_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Dataset save to {data_path}.")

    return data_path


if __name__ == "__main__":
    data_args, = get_prepare_args(sys.argv[1])
    prepare(data_args)
