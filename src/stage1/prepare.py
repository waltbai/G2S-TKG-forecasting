import json
import logging
import os

from typing import List, Dict, Any

from transformers import HfArgumentParser

from src.utils.anonymizer import get_anonymizer
from src.stage1.prompt import get_prompt_constructor
from src.utils.args import AnonymizedDataArguments
from src.utils.history_finder import get_history_finder
from src.utils.time_processor import get_time_processor
from src.utils.query import Query, construct_queries
from src.utils.tkg import TKG

logger = logging.getLogger(__name__)


def get_data_version(args: AnonymizedDataArguments):
    """Get data name."""
    name = "stage1"
    if args.train_dataset:
        name += "-train_" + args.train_dataset.replace(",", "_")
    if args.valid_dataset:
        name += "-valid_" + args.valid_dataset.replace(",", "_")
    if args.test_dataset:
        name += "-test_" + args.test_dataset.replace(",", "_")
    name += (
        f"-{args.history_finder}"
        f"-{args.history_type}"
        f"-{args.history_direction}"
        f"-{args.history_length}"
        f"-{args.anonymizer}"
    )
    if args.anonymize_entity:
        name += "-anony_ent"
    if args.anonymize_rel:
        name += "-anony_rel"
    name += f"-{args.time_processor}"
    name += f"-{args.prompt_construct_strategy}"
    return name


def convert_queries(queries: List[Query]) -> Dict[str, Any]:
    """Convert queries into dataset format."""
    converted_queries = {
        "prompt": [query.prompt for query in queries],
        "label": [query.label for query in queries],
        "filters": [query.anonymous_filters for query in queries],
        "candidates": [
            {v: k for k, v in query.entity_mapping.items()}
            for query in queries
        ],
    }
    return converted_queries


def prepare(
        data_args: AnonymizedDataArguments,
        use_tqdm: bool = True,
) -> str:
    """Main data preparation function for stage-1."""
    # Prepare variables and objects
    prepare_dir = data_args.prepare_dir
    dataset_dir = data_args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)
    datafile_name = get_data_version(data_args) + ".json"
    data_path = os.path.join(prepare_dir, datafile_name)
    if os.path.exists(data_path):
        if not data_args.overwrite_cache:
            logger.info(f"Dataset {data_path} exists. Skip prepare step")
            return data_path
        else:
            logger.info(f"Dataset {data_path} exists. Overwrite existing ones")
    else:
        logger.info(f"Prepare dataset")

    # Get strategies
    history_finder = get_history_finder(
        history_finder=data_args.history_finder,
        history_type=data_args.history_type,
        history_direction=data_args.history_direction,
        history_length=data_args.history_length,
        use_tqdm=use_tqdm,
    )
    anonymizer = get_anonymizer(
        strategy=data_args.anonymizer,
        anonymize_entity=data_args.anonymize_entity,
        anonymize_rel=data_args.anonymize_rel,
        use_tqdm=use_tqdm,
    )
    time_processor = get_time_processor(
        strategy=data_args.time_processor,
        use_tqdm=use_tqdm,
    )
    prompt_constructor = get_prompt_constructor(
        strategy=data_args.prompt_construct_strategy,
        use_tqdm=use_tqdm,
    )

    # Prepare datasets
    train_datasets = [dataset for dataset in data_args.train_dataset.split(",") if dataset]
    valid_datasets = [dataset for dataset in data_args.valid_dataset.split(",") if dataset]
    test_datasets = [dataset for dataset in data_args.test_dataset.split(",") if dataset]
    datasets = list(set(train_datasets + valid_datasets + test_datasets))
    train_queries, valid_queries, test_queries = [], [], []
    for dataset in datasets:
        # Load TKG
        logger.info(f"Loading TKG {dataset}")
        tkg = TKG.load(dataset_dir, dataset)

        # Construct queries
        logger.info("Construct queries")
        if dataset in train_datasets:
            temp_train_queries = construct_queries(
                tkg.train_facts,
                use_tqdm=use_tqdm,
            )
        else:
            temp_train_queries = []
        if dataset in valid_datasets:
            temp_valid_queries = construct_queries(
                tkg.valid_facts,
                use_tqdm=use_tqdm,
            )
        else:
            temp_valid_queries = []
        if dataset in test_datasets:
            temp_test_queries = construct_queries(
                tkg.test_facts,
                use_tqdm=use_tqdm,
            )
        else:
            temp_test_queries = []
        queries = temp_train_queries + temp_valid_queries + temp_test_queries

        # Construct history
        logger.info("Find historical facts")
        history_finder(
            queries=queries,
            tkg=tkg,
        )

        # Anonymize queries
        logger.info("Anonymize queries")
        anonymizer(
            queries=queries,
            tkg=tkg,
        )

        # Process time
        logger.info("Process time")
        time_processor(
            queries=queries,
            tkg=tkg,
        )

        # Construct anonymized prompt
        logger.info("Construct anonymized prompt")
        prompt_constructor(queries=queries)

        # Split queries and append
        train_queries.extend(temp_train_queries)
        valid_queries.extend(temp_valid_queries)
        test_queries.extend(temp_test_queries)

    # Save data
    data = {
        "train": convert_queries(train_queries),
        "valid": convert_queries(valid_queries),
        "test": convert_queries(test_queries),
    }
    with open(data_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Dataset save to {data_path}")
    return data_path


if __name__ == "__main__":
    # Parse arguments from config file
    parser = HfArgumentParser([AnonymizedDataArguments])
    data_args, = parser.parse_args_into_dataclasses()
    # Prepare
    prepare(data_args)
