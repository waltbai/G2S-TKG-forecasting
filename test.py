import argparse

from llm4tkg.model.icl_model import InContextLearningModel
from llm4tkg.preprocess.tkg import TemporalKG


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        type=str, default="ICEWS14s",
                        choices=["ICEWS14s", "ICEWS18", "WIKI", "YAGO", "GDELT"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tkg = TemporalKG.load(args.dataset)
    model = InContextLearningModel(
        backbone="/data/bailong/models/gpt2",
        device="cuda:0",
    )
    model.evaluate(tkg)
