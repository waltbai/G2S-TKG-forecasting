import logging
import os
import sys

from src.old.stage2.args import get_prepare_args
from src.old.stage2.prompt import PromptConstructor
from src.tkg import TKG

logger = logging.getLogger(__name__)


def main():
    """
    Prepare dataset.
    """
    data_args, = get_prepare_args(sys.argv[1])
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
                tkg.find_history(
                    query=query,
                    strategy=data_args.history_strategy,
                    history_length=data_args.history_length,
                )
        # Construct prompts
        for part in ["train", "valid", "test"]:
            logger.info(f"Construct {dataset} {part} prompts.")
            for query in tmp_queries[part]:
                prompt_func(
                    query=query,
                    tkg=tkg,
                    map_strategy=data_args.map_strategy,
                    time_strategy=data_args.time,
                    map_entity=data_args.entity,
                    map_relation=data_args.relation,
                )
        for part in ["train", "valid", "test"]:
            queries[part].extend(tmp_queries[part])

    test_queries = queries["test"]
    hit, total = 0, 0
    tot_len = 0
    for query in test_queries:
        label = query.label
        ids = set()
        for fact in query.history:
            ids.add(query.entity_mapping[fact.head])
            ids.add(query.entity_mapping[fact.tail])
        ids.add(query.entity_mapping[query.entity])
        if label in ids:
            hit += 1
        total += 1
        tot_len += len(query.history)
    logger.info(f"{hit} out of {total} ({hit/total:.2%}) queries has hit the answer.")
    logger.info(f"Average history length: {tot_len/total:.2f}")


if __name__ == "__main__":
    main()
