import argparse
import logging

from src.model.icl_model import InContextLearningModel
from src.preprocess.tkg import TemporalKG
from src.prompt import quadruple_prompt


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    # Experiment settings
    parser.add_argument("--dataset",
                        type=str, default="ICEWS14s",
                        choices=["ICEWS14s", "ICEWS18", "WIKI", "YAGO", "GDELT"])
    parser.add_argument("--history_type",
                        type=str, default="entity",
                        choices=["entity", "pair"])
    parser.add_argument("--history_direction",
                        type=str, default="uni",
                        choices=["uni", "bi"])
    parser.add_argument("--anonymize_entity",
                        type=str, default=None)
    parser.add_argument("--anonymize_rel",
                        type=str, default=None)
    parser.add_argument("--anonymize_time",
                        type=str, default="index")
    parser.add_argument("--predictions",
                        type=int, default=30)
    parser.add_argument("--time_filter",
                        type=bool, default=False)
    # Basic settings
    parser.add_argument("--pbar", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tkg = TemporalKG.load(
        args.dataset,
        anonymize_entity=args.anonymize_entity,
        anonymize_rel=args.anonymize_rel,
        anonymize_time=args.anonymize_time,
    )
    model = InContextLearningModel(
        backbone="/data/bailong/models/gpt2",
        device="cuda:0",
        history_type=args.history_type,
        history_direction=args.history_direction,
        time_filter=args.time_filter,
        pbar=args.pbar,
    )
    model.evaluate(tkg)
