from dataclasses import dataclass
from typing import List, Dict


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
    # Predictions
    prompt: str = ""
    candidates: Dict[str, str] = None
    predictions: List[str] = None
    scores: List[float] = None

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
