from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple

from llm4tkg.preprocess.fact import Fact


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


def metric(results: List[QueryResult]) -> Dict[str, float]:
    """Compute metrics."""
    hit1, hit3, hit10, total = 0, 0, 0, 0
    for result in results:
        for rank, pred in enumerate(result.predictions):
            if pred.entity == result.answer:
                if 0 <= rank < 1:
                    hit1 += 1
                if 0 <= rank < 3:
                    hit3 += 1
                if 0 <= rank < 10:
                    hit10 += 1
        total += 1
    hit1 = hit1 / total
    hit3 = hit3 / total
    hit10 = hit10 / total
    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@10": hit10,
    }
