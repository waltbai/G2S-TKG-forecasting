import logging
import os

from src.args import AnonymizedDataArguments
from src.utils.tkg import TKG

logger = logging.getLogger(__name__)



def get_stage1_data_name(args: AnonymizedDataArguments):
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
        name += "-prefix"
    if args.anonymize_strategy == "session":
        name += f"-{args.anonymize_base_time}"
    return name


def prepare(args: AnonymizedDataArguments):
    """Main data preparation function for stage-1."""
    prepare_dir = args.prepare_dir
    dataset_dir = args.dataset_dir
    os.makedirs(prepare_dir, exist_ok=True)

    # Prepare datasets
    train_datasets = [_ for _ in args.train_dataset.split(",") if _]
    valid_datasets = [_ for _ in args.valid_dataset.split(",") if _]
    test_datasets = [_ for _ in args.test_dataset.split(",") if _]
    datasets = list(set(train_datasets + valid_datasets + test_datasets))
    for dataset in datasets:
        # Load TKG
        logger.info(f"Loading TKG {dataset}.")
        tkg = TKG.load(dataset_dir, dataset)

        # Construct queries
        logger.info("Construct queries.")
        
