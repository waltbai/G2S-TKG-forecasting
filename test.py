from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.utils.config import load_config

if __name__ == "__main__":
    tkg = TemporalKG.load("GDELT")
