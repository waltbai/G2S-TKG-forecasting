import random

from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.utils.config import load_config

if __name__ == "__main__":
    tkg = TemporalKG.load("ICEWS14s")
    query = random.choice(tkg.train_set)
    print(f"Query: ({query.head}, {query.rel}, ?, {query.time})")
    print()
    print("Prompt:")
    print(tkg.construct_prompt(query, anonymous=False))
