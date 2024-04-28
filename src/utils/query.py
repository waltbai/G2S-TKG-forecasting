from dataclasses import dataclass
from datetime import datetime

from typing import List, Dict


@dataclass
class Query:
    """Query class."""
    entity: str
    rel: str
    answer: str
    time: str
    # Direction
    entity_role: str
    # Filter answers
    filters: List[str] = None
    # Predictions
    prompt: str = ""
    candidates: Dict[str, str] = None
    predictions: List[str] = None
    scores: List[float] = None
