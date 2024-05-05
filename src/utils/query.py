from dataclasses import dataclass

from typing import List


@dataclass
class Query:
    """Query class.

    Args:
        entity (str): query entity.
        rel (str): query relation.
        answer (str): answer entity.
        time (str): query time.
        entity_role (str):
            role type of query entity,
            should be either "head" or "tail".
        filters (List[str]):
            entities that should be filtered in time-filter metrics.
    """
    entity: str
    rel: str
    answer: str
    time: str
    entity_role: str = "head"
    filters: List[str] = None
