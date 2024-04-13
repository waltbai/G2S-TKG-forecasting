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
        query_target: str = "tail",
) -> bool:
    """Check if the fact is needed.

    For query fact (S, R, O, T),
    if query_target == "tail", then the query is (S, R, ?, T),
    otherwise the query is (?, R, O, T).

    For query (?, R, O, T), when history_direction is "uni",
    we search history (?, ?, O, <T), which maintain is original order.
    """
    flag = True
    if history_direction == "uni":
        check_entity = fact.head if query_target == "tail" else fact.tail
        flag = flag and query_entity == check_entity
    else:   # history_direction == "bi"
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
) -> Tuple[str, Dict[str, str]]:
    """Construct quadruple-like prompt.

    This prompt is used in Lee, et al., emnlp, 2023.
    """
    # Find history
    history = []
    query_entity = query.head if query_target == "tail" else query.tail
    for fact in facts:
        if _check_fact(
                fact=fact,
                query_entity=query_entity,
                query_relation=query.rel,
                query_time=query.time,
                history_type=history_type,
                history_direction=history_direction,
                query_target=query_target,
        ):
            history.append(
                fact.prompt_quadruple(
                    query_entity=query_entity,
                    anonymous=anonymous,
                    anonymous_time=anonymous_time,
                )
            )
    if history_length is not None:
        history = history[-history_length:]
    # shuffle history
    if shuffle:
        history = random.sample(history, len(history))
    # count tail frequency
    candidate_freq = {}
    for fact in history:
        candidate_freq.setdefault(fact[2], 0)
        candidate_freq[fact[2]] += 1
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
        head, rel, tail, time = fact
        task_input += f"{time}:[{head},{rel},{candidate_mapping[tail]}.{tail}]\n"
    # Append query
    head, rel, tail, time = query.prompt_quadruple(
        query_entity=query_entity,
        anonymous=anonymous,
        anonymous_time=anonymous_time,
    )
    task_input += f"{time}:[{head},{rel},"
    # Prepare candidates:
    candidates = {str(v): k for k, v in candidate_mapping.items()}
    return task_input, candidates
