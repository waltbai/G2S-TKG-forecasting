from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple

from src.preprocess.fact import Fact
from src.utils.common import card2ord


@dataclass
class Prediction:
    """Prediction class."""
    entity: str
    score: float


@dataclass
class QueryResult:
    """Result of a query."""
    query: Fact
    query_target: str
    predictions: List[Prediction]
    candidates: Dict[str, str]
    prompt: str = ""
    answer: str = ""

    def time_filter(
            self,
            time_filter_set: Set[Tuple[str, str, str, str]],
    ) -> None:
        """In-place filter valid predictions."""
        filtered = []
        for item in self.predictions:
            if self.query_target == "tail":
                quad = (
                    self.query.head,
                    self.query.rel,
                    item.entity,
                    self.query.time,
                )
            else:
                quad = (
                    item.entity,
                    self.query.rel,
                    self.query.tail,
                    self.query.time,
                )
            if item.entity != self.answer and quad in time_filter_set:
                continue
            filtered.append(item)
        self.predictions = filtered


@dataclass
class Query:
    """Universal class for query."""
    # Strings
    entity: str
    rel: str
    answers: List[str]
    time: str
    # Direction
    direction: str
    # Indices
    entity_idx: int = None
    rel_idx: int = None
    answers_idx: List[int] = None
    time_idx: int = None
    # Predictions
    prompt: str = ""
    candidates: Dict[str, str] = None
    predictions: List[str] = None
    scores: List[float] = None

    def prompt_quadruple(
            self,
            anonymize: bool = False,
            anonymize_time: bool = True,
    ) -> Tuple[str, str, str]:
        """Representation for quadruple prompt."""
        if anonymize:
            entity, rel = self.entity_idx, self.rel_idx
        else:
            entity, rel = self.entity, self.rel
        if anonymize_time:
            time = self.time_idx
        else:
            time = self.time
        entity, rel, time = map(str, (entity, rel, time))
        return entity, rel, time

    def prompt_text(
            self,
            anonymize: bool = False,
            anonymize_time: bool = True,
    ) -> Tuple[str, str, str]:
        """Representation for text prompt."""
        if anonymize:
            entity, rel = self.entity_idx, self.rel_idx
        else:
            entity, rel = self.entity, self.rel
        if anonymize_time:
            time = card2ord(self.time_idx)
        else:
            time = self.time
        entity, rel, time = map(str, (entity, rel, time))
        return entity, rel, time


def metric(
        queries: List[Query],
        time_filter: bool = True
) -> Dict[str, float]:
    """Compute raw metrics."""
    hit1, hit3, hit10, total = 0, 0, 0, 0
    for query in queries:
        for answer in query.answers:
            try:
                idx = query.predictions.index(answer)
                # If time-filter metrics are applied,
                # check previous predictions
                if time_filter:
                    for pred in query.predictions[:idx]:
                        if pred in query.answers:
                            idx -= 1
                # Calculate hits
                if idx < 1:
                    hit1 += 1
                if idx < 3:
                    hit3 += 1
                if idx < 10:
                    hit10 += 1
            except ValueError:
                # Not in the list
                pass
        total += len(query.answers)
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@10": hit10 / total,
    }
