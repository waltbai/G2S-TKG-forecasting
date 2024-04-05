import random

from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.utils.config import load_config

if __name__ == "__main__":
    tkg = TemporalKG.load("ICEWS18")
    # query = random.choice(tkg.train_set)
    query = tkg.test_set[0]
    print(f"Query: ({query.head}, {query.rel}, ?, {query.time})")
    print()
    task_input, candidates = tkg.construct_prompt(
        query,
        anonymous=False,
        anonymous_time=True,
        label=True,
        shuffle=False,
    )
    print("Input:")
    print(task_input)
    print("Candidates:")
    print(candidates)
