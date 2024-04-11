import random
from typing import List, Dict, Tuple

from llm4tkg.preprocess.fact import Fact


def _check_fact(
        fact: Fact,
        query_entity: str,
        query_relation: str,
        query_time: str,
        history_type: str = "entity",
        history_direction: str = "uni",
) -> bool:
    """Check if the fact is needed."""
    flag = True
    if history_direction == "uni":
        flag = flag and query_entity == fact.head
    else:   # bi
        flag = flag and (query_entity == fact.head or query_entity == fact.tail)
    if history_type == "pair":
        flag = flag and query_relation == fact.rel
    flag = flag and (fact.time < query_time)
    return flag


def quadruple_prompt(
        query: Fact,
        facts: List[Fact],
        history_length: int = 30,
        history_type: str = "entity",
        history_direction: str = "uni",
        anonymous: bool = False,
        anonymous_time: bool = True,
        shuffle: bool = False,
        query_target: str = "tail",
) -> Tuple[str, Dict[int | str, str]]:
    """Construct quadruple-like prompt.

    This prompt is used in Lee, et al., emnlp, 2023.
    """
    # Find history
    history = []
    query_entity = query.head if query_target == "tail" else query.tail
    query_entity_idx = query.head_idx if query_target == "tail" else query.tail_idx
    for fact in facts:
        if _check_fact(
                fact=fact,
                query_entity=query_entity,
                query_relation=query.rel,
                query_time=query.time,
                history_type=history_type,
                history_direction=history_direction,
        ):
            history.append(fact)
    if history_length is not None:
        history = history[-history_length:]
    # shuffle history
    if shuffle:
        history = random.sample(history, len(history))
    # count tail frequency
    candidate_freq = {}
    for fact in history:
        if query_entity == fact.head:
            candidate_entity = fact.tail
        else:
            candidate_entity = fact.head
        candidate_freq.setdefault(candidate_entity, 0)
        candidate_freq[candidate_entity] += 1
    candidate_sorted = list(
        sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
    )
    # candidate mapping to index, start from 0
    candidate_mapping = {}
    for i, (entity, _) in enumerate(candidate_sorted):
        candidate_mapping.setdefault(entity, i)
    # Append historical facts
    task_input = ""
    for fact in history:
        time_repr = fact.time_idx if anonymous_time else fact.time
        head_repr = fact.head_idx if anonymous else fact.head
        rel_repr = fact.rel_idx if anonymous else fact.rel
        tail_repr = fact.tail_idx if anonymous else fact.tail
        # Check if tail == query_entity, default to use swap strategy.
        if fact.tail == query_entity:
            head_repr, tail_repr = tail_repr, head_repr
        tail_repr = f"{candidate_mapping[fact.tail]}.{tail_repr}"
        task_input += f"{time_repr}:[{head_repr},{rel_repr},{tail_repr}]\n"
    # Append query
    time_repr = query.time_idx if anonymous_time else query.time
    ent_repr = query_entity_idx if anonymous else query_entity
    rel_repr = query.rel_idx if anonymous else query.rel
    task_input += f"{time_repr}:[{ent_repr},{rel_repr},"
    # Prepare candidates:
    candidates = {v: k for k, v in candidate_mapping.items()}
    return task_input, candidates
