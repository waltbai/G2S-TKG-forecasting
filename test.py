from llm4tkg.model.icl_model import InContextLearningModel
from llm4tkg.preprocess.tkg import TemporalKG


if __name__ == "__main__":
    tkg = TemporalKG.load("ICEWS18")
    model = InContextLearningModel(
        backbone="/data/bailong/models/gpt2",
        device="cuda:0"
    )
    model.evaluate(tkg)
