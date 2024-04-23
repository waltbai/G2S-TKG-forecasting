import random
from typing import List, Dict, Tuple

from src.evaluation import Query
from src.preprocess.fact import Fact
from src.preprocess.tkg import TemporalKG


def _check_fact(
        fact: Fact,
        query: Query,
        history_type: str = "entity",
        history_direction: str = "uni",
) -> bool:
    """Check if the fact is needed.

    For query fact (S, R, O, T),
    if query.direction == "tail", then the query is (S, R, ?, T),
    otherwise the query is (?, R, O, T).

    For query (?, R, O, T), when history_direction is "uni",
    we search history (?, ?, O, <T), which maintain is original order.
    """
    flag = True
    if history_direction == "uni":
        check_entity = fact.head if query.direction == "tail" else fact.tail
        flag = flag and query.entity == check_entity
    else:   # history_direction == "bi"
        flag = flag and (query.entity == fact.head or query.entity == fact.tail)
    if history_type == "pair":
        flag = flag and query.rel == fact.rel
    flag = flag and (fact.time < query.time)
    return flag


def quadruple_prompt(
        query: Query,
        tkg: TemporalKG,
        history_length: int = 30,
        history_type: str = "entity",
        history_direction: str = "uni",
        shuffle: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """Construct quadruple-like prompt.

    This prompt is used in Lee, et al., emnlp, 2023.
    """
    # Find history
    history = []
    if history_direction == "bi":
        facts = tkg.find_history_by_both(query.entity)
    elif query.direction == "tail":
        facts = tkg.find_history_by_head(query.entity)
    else:   # query.direction == "head"
        facts = tkg.find_history_by_tail(query.entity)
    for fact in facts:
        if _check_fact(
                fact=fact,
                query=query,
                history_type=history_type,
                history_direction=history_direction,
        ):
            if query.entity == fact.tail:
                head, rel, tail, time = fact.quadruple("swap")
            else:
                head, rel, tail, time = fact.quadruple()
            head = tkg.anonymize_entity(head)
            rel = tkg.anonymize_rel(rel)
            tail = tkg.anonymize_entity(tail)
            time = tkg.anonymize_time(time)
            history.append((head, rel, tail, time))
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
    entity = tkg.anonymize_entity(query.entity)
    rel = tkg.anonymize_rel(query.rel)
    time = tkg.anonymize_time(query.time)
    task_input += f"{time}:[{entity},{rel},"
    # Prepare candidates:
    candidates = {str(v): k for k, v in candidate_mapping.items()}
    return task_input, candidates
