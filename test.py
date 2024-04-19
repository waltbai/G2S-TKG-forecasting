import argparse

from src.model.icl_model import InContextLearningModel
from src.preprocess.tkg import TemporalKG
from src.prompt import quadruple_prompt


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        type=str, default="ICEWS14s",
                        choices=["ICEWS14s", "ICEWS18", "WIKI", "YAGO", "GDELT"])
    parser.add_argument("--history_type",
                        type=str, default="entity",
                        choices=["entity", "pair"])
    parser.add_argument("--history_direction",
                        type=str, default="uni",
                        choices=["uni", "bi"])
    parser.add_argument("--anonymize",
                        type=bool, default=False)
    parser.add_argument("--anonymize_time",
                        type=bool, default=True)
    parser.add_argument("-k", "--top_k",
                        type=int, default=30)
    parser.add_argument("--time_filter",
                        type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tkg = TemporalKG.load(args.dataset)
    model = InContextLearningModel(
        backbone="/data/bailong/models/gpt2",
        device="cuda:0",
        history_type=args.history_type,
        history_direction=args.history_direction,
        anonymize=args.anonymize,
        anonymize_time=args.anonymize_time,
        time_filter=args.time_filter,
    )
    model.evaluate(tkg)
