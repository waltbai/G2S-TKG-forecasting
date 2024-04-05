import random
from typing import List, Dict, Tuple

from llm4tkg.preprocess.fact import Fact


def quadruple_prompt(
        query: Fact,
        history: List[Fact],
        anonymous: bool = False,
        anonymous_time: bool = True,
        label: bool = False,
        shuffle: bool = False,
) -> Tuple[str, Dict[int | str, str]]:
    """Construct quadruple-like prompt.

    This prompt is used in Lee, et al., emnlp, 2023.
    """
    # shuffle history
    if shuffle:
        history = random.sample(history, len(history))
    # count tail frequency
    candidate_freq = {}
    for fact in history:
        candidate_freq.setdefault(fact.tail, 0)
        candidate_freq[fact.tail] += 1
    candidate_sorted = list(
        sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
    )
    # candidate mapping to index, start from 0
    candidate_mapping = {}
    for i, (tail, _) in enumerate(candidate_sorted):
        candidate_mapping.setdefault(tail, i)
    # Append historical facts
    task_input = ""
    for fact in history:
        time_repr = fact.time_idx if anonymous_time else fact.time
        head_repr = fact.head_idx if anonymous else fact.head
        rel_repr = fact.rel_idx if anonymous else fact.rel
        tail_repr = fact.tail_idx if anonymous else fact.tail
        if label:
            tail_repr = f"{candidate_mapping[fact.tail]}.{tail_repr}"
        task_input += f"{time_repr}:[{head_repr},{rel_repr},{tail_repr}]\n"
    # Append query
    time_repr = query.time_idx if anonymous_time else query.time
    head_repr = query.head_idx if anonymous else query.head
    rel_repr = query.rel_idx if anonymous else query.rel
    task_input += f"{time_repr}:[{head_repr},{rel_repr},"
    # Prepare candidates:
    if label:
        candidates = {v: k for k, v in candidate_mapping.items()}
    else:
        candidates = {k: k for k, _ in candidate_mapping.items()}
    return task_input, candidates
