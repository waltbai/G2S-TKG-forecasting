from dataclasses import dataclass

from typing import Tuple


@dataclass
class Fact:
    """
    Fact class.
    """
    head: str
    rel: str
    tail: str
    time: int

    def quadruple(self) -> Tuple[str, str, str, int]:
        """
        Quadruple representation of the fact.
        """
        return self.head, self.rel, self.tail, self.time
