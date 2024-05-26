from dataclasses import dataclass

from typing import List, Dict, Tuple


@dataclass
class Fact:
    """Temporal fact class."""
    head: str
    rel: str
    tail: str
    time: str

    @classmethod
    def from_ids(
            cls,
            ids: List[int],
            entities: List[str],
            relations: List[str],
            id2time: Dict[int, str],
    ):
        """Convert ids to fact."""
        assert len(ids) >= 4, "Unable to unpack `ids` to fact!"
        # Default ids in order: (head_id, rel_id, tail_id, time_id)
        head = entities[ids[0]]
        rel = relations[ids[1]]
        tail = entities[ids[2]]
        time = id2time[ids[3]]
        return cls(head, rel, tail, time)

    def quadruple(
            self,
            _format: str = "normal",
    ) -> Tuple[str, str, str, str]:
        """Quadruple representation.

        Args:
            _format (str): select from normal/inverse/swap.

        Returns:
            Quadruple: (head, rel, tail, time)
        """
        head = self.head
        rel = self.rel
        tail = self.tail
        time = self.time
        if _format != "normal":
            head, tail = tail, head
        if _format == "inverse":
            rel = f"inverse {rel}"
        return head, rel, tail, time
