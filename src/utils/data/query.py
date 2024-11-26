from dataclasses import dataclass, field

from typing import List, Dict, Tuple, Any

from src.utils.data.fact import Fact


@dataclass
class Query:
    """
    Query class.
    """
    # Initialize Variables
    entity: str
    rel: str
    answer: str
    time: int
    role: str
    # Processed Variables
    history: List[Fact] = field(default_factory=list)
    entity_mapping: Dict[str, str] = field(default_factory=dict)
    rel_mapping: Dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    label: str = ""
    filters: List[str] = field(default_factory=list)

    @classmethod
    def from_fact(cls, fact: Fact, role: str):
        """Construct query from fact."""
        rel = fact.rel
        time = fact.time
        if role == "head":
            entity, answer = fact.head, fact.tail
        elif role == "tail":
            entity, answer = fact.tail, fact.head
        else:
            raise ValueError(f"Unknown role: {role}")
        return cls(
            entity=entity,
            rel=rel,
            answer=answer,
            time=time,
            role=role,
        )

    def key(self) -> Tuple[str, str, int, str]:
        """
        Generate a search key for query, including its basic elements.
        """
        return self.entity, self.rel, self.time, self.role

    def dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary representation of query.
        """
        return {
            "prompt": self.prompt,
            "label": self.label,
            "filters": self.filters,
            "id2entity": {v: k for k, v in self.entity_mapping.items()},
        }
