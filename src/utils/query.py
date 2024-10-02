from dataclasses import dataclass, field

from typing import Dict, List

from tqdm import tqdm

from src.utils.fact import Fact


@dataclass
class Query:
    """Query class."""
    entity: str
    rel: str
    answer: str
    time: str
    entity_role: str = "head"
    # Processed data
    history: List[Fact] = field(default_factory=list)
    # Mapping from real name to anonymous ones
    entity_mapping: Dict[str, str] = field(default_factory=dict)
    rel_mapping: Dict[str, str] = field(default_factory=dict)
    # Mapping from the absolute time to processed time
    time_mapping: Dict[str, str] = field(default_factory=dict)
    # Data for Prediction Models
    prompt: str = ""
    label: str = ""
    filters: List[str] = field(default_factory=list)
    anonymous_filters: List[str] = field(default_factory=list)
    candidates: Dict[str, str] = field(default_factory=dict)


def construct_queries(
        facts: List[Fact],
        predict_head: bool = True,
        use_tqdm: bool = False,
) -> List[Query]:
    """
    Construct queries.

    Args:
        facts (List[Fact]):
            List of facts to construct queries.
        predict_head (bool):
            Whether to predict head or not.
            Defaults to True.
        use_tqdm (bool):
            Whether to use tqdm or not.
            Defaults to False.

    Returns:
        List[Query]:
            List of constructed queries.
    """
    queries = []
    answers = {}
    roles = ["head", "tail"] if predict_head else ["head"]
    total = len(roles) * len(facts)
    with tqdm(total=total, disable=not use_tqdm) as pbar:
        for entity_role in roles:
            for fact in facts:
                query_rel = fact.rel
                query_time = fact.time
                if entity_role == "head":
                    query_entity = fact.head
                    answer = fact.tail
                else:  # entity_role == "tail"
                    query_entity = fact.tail
                    answer = fact.head
                key = (query_entity, query_rel, query_time, entity_role)
                answers.setdefault(key, []).append(answer)
                queries.append(
                    Query(
                        entity=query_entity,
                        rel=query_rel,
                        answer=answer,
                        time=query_time,
                        entity_role=entity_role,
                    )
                )
                pbar.update()

    # Sort queries by time
    for query in queries:
        key = (query.entity, query.rel, query.time, query.entity_role)
        query.filters = [_ for _ in answers[key] if _ != query.answer]
    queries = sorted(queries, key=lambda x: x.time)
    return queries
