from dataclasses import dataclass, field

from typing import Dict, List

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
