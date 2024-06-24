import json
import logging
import os
import sys

from typing import List, Dict, Any

from tqdm import tqdm
from transformers import HfArgumentParser

from src.args import AnonymizedDataArguments
from src.utils.anonymizer import get_anonymizer
from src.stage1.prompt import get_prompt_constructor
from src.utils.time_processor import get_time_processor
from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import TKG

logger = logging.getLogger(__name__)


def construct_queries(
        facts: List[Fact],
        predict_head: bool = True,
        use_tqdm: bool = False,
) -> List[Query]:
    """
    Construct queries.

    Args:
        facts (List[Fact]):
            List of facts to construct queries.
        predict_head (bool):
            Whether to predict head or not.
            Defaults to True.
        use_tqdm (bool):
            Whether to use tqdm or not.
            Defaults to False.

    Returns:
        List[AnonymizedQuery]:
            List of constructed queries.
    """
    queries = []
    answers = {}
    roles = ["head", "tail"] if predict_head else ["head"]
    total = len(roles) * len(facts)
    with tqdm(total=total, disable=not use_tqdm) as pbar:
        for entity_role in roles:
            for fact in facts:
                query_rel = fact.rel
                query_time = fact.time
                if entity_role == "head":
                    query_entity = fact.head
                    answer = fact.tail
                else:  # entity_role == "tail"
                    query_entity = fact.tail
                    answer = fact.head
                key = (query_entity, query_rel, query_time, entity_role)
                answers.setdefault(key, []).append(answer)
                queries.append(
                    Query(
                        entity=query_entity,
                        rel=query_rel,
                        answer=answer,
                        time=query_time,
                        entity_role=entity_role,
                    )
                )
                pbar.update()

    # Sort queries by time
    for query in queries:
        key = (query.entity, query.rel, query.time, query.entity_role)
        query.filters = [_ for _ in answers[key] if _ != query.answer]
    queries = sorted(queries, key=lambda x: x.time)
    return queries


def construct_history(
        queries: List[Query],
        tkg: TKG,
        history_length: int = 30,
        history_type: str = "entity",
        history_direction: str = "uni",
        use_tqdm: bool = False,
) -> None:
    """
    Construct history for each query.

    Args:
        queries (List[AnonymizedQuery]):
            List of queries.
        tkg (TKG):
            TKG to search for history.
        history_length (int):
            Length of history facts.
            Defaults to 30.
        history_type (str):
            Type of matching facts, either by "entity" or "pair".
            Defaults to "entity".
        history_direction (str):
            Direction of matching facts, either "uni" or "bi" direction.
            Defaults to "uni".
        use_tqdm (bool):
            Whether to use tqdm or not.
            Defaults to False.
    """
    with tqdm(total=len(queries), disable=not use_tqdm) as pbar:
        for query in queries:
            # Search facts
            if history_direction == "uni":
                if history_type == "entity":
                    search_key_1 = query.entity_role
                    search_key_2 = query.entity
                else:  # history_type == "pair"
                    search_key_1 = f"{query.entity_role}_rel"
                    search_key_2 = (query.entity, query.rel)
            else:  # history_direction == "bi"
                if history_type == "entity":
                    search_key_1 = "both"
                    search_key_2 = query.entity
                else:  # history_type == "pair"
                    search_key_1 = "both_rel"
                    search_key_2 = (query.entity, query.rel)
            facts = tkg.search_history[search_key_1][search_key_2]

            # Filter future facts and cut-off
            history = [fact for fact in facts if fact.time < query.time]
            history = history[-history_length:]
            query.history = history

            pbar.update()


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
        f"-{args.history_type}"
        f"-{args.history_direction}"
        f"-{args.history_length}"
        f"-{args.anonymize_strategy}"
    )
    if args.anonymize_prefix:
        name += "_prefix"
    name += f"-{args.time_process_strategy}"
    if args.vague_time:
        name += "_vague"
    name += f"-{args.prompt_construct_strategy}"
    if args.candidate_relabel:
        name += f"-relabel"
    return name


def convert_queries(queries: List[Query]) -> Dict[str, Any]:
    """Convert queries into dataset format."""
    converted_queries = {
        "prompt": [query.prompt for query in queries],
        "label": [query.label for query in queries],
        "filters": [query.anonymous_filters for query in queries],
        "candidates": [query.candidates for query in queries],
    }
    return converted_queries


def prepare(
        args: AnonymizedDataArguments,
        use_tqdm: bool = True,
) -> str:
    """Main data preparation function for stage-1."""
    # Prepare variables and objects
    prepare_dir = args.prepare_dir
    dataset_dir = args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)
    datafile_name = get_data_version(args) + ".json"
    data_path = os.path.join(prepare_dir, datafile_name)
    if os.path.exists(data_path):
        if not args.overwrite_cache:
            logger.info(f"Dataset {data_path} exists. Skip prepare step")
            return data_path
        else:
            logger.info(f"Dataset {data_path} exists. Overwrite existing ones")
    else:
        logger.info(f"Prepare dataset")

    # Get strategies
    anonymizer = get_anonymizer(
        strategy=args.anonymize_strategy,
        use_tqdm=use_tqdm,
    )
    time_processor = get_time_processor(
        strategy=args.time_process_strategy,
        use_tqdm=use_tqdm,
    )
    prompt_constructor = get_prompt_constructor(
        strategy=args.prompt_construct_strategy,
        prefix=args.anonymize_prefix,
        cand_relabel=args.candidate_relabel,
        use_tqdm=use_tqdm,
    )

    # Prepare datasets
    train_datasets = [dataset for dataset in args.train_dataset.split(",") if dataset]
    valid_datasets = [dataset for dataset in args.valid_dataset.split(",") if dataset]
    test_datasets = [dataset for dataset in args.test_dataset.split(",") if dataset]
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
        # num_train = len(temp_train_queries)
        # num_valid = len(temp_valid_queries)
        # num_test = len(temp_test_queries)
        queries = temp_train_queries + temp_valid_queries + temp_test_queries

        # Construct history
        logger.info("Construct history")
        construct_history(
            queries=queries,
            tkg=tkg,
            history_length=args.history_length,
            history_type=args.history_type,
            history_direction=args.history_direction,
            use_tqdm=use_tqdm,
        )

        # Anonymize queries
        logger.info("Anonymize queries")
        anonymizer(
            queries=queries,
            tkg=tkg,
            prefix=args.anonymize_prefix
        )

        # Process time
        logger.info("Process time")
        time_processor(
            queries=queries,
            tkg=tkg,
            vague=args.vague_time
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
    config_path = sys.argv[1]
    parser = HfArgumentParser([AnonymizedDataArguments])
    data_args, = parser.parse_yaml_file(os.path.abspath(config_path))
    # Prepare
    prepare(data_args)
