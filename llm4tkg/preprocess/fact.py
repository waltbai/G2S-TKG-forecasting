from pydantic import BaseModel
from typing import Tuple


class Fact(BaseModel):
    # Fact part
    head: str
    rel: str
    tail: str
    time: str
    # Index part
    head_idx: int = None
    rel_idx: int = None
    tail_idx: int = None
    time_idx: int = None

    def quadruple(
            self,
            reverse: bool = False
    ) -> Tuple[str, str, str, str]:
        """Quadruple representation."""
        if reverse:
            return (
                self.tail,
                f"inverse {self.rel}",
                self.head,
                self.time,
            )
        else:
            return (
                self.head,
                self.rel,
                self.tail,
                self.time,
            )

    def quadruple_idx(self) -> Tuple[int, int, int, int]:
        """Quadruple index representation."""
        return (
            self.head_idx,
            self.rel_idx,
            self.tail_idx,
            self.time_idx,
        )

    def __str__(self):
        return (f"({self.head},"
                f" {self.rel},"
                f" {self.tail},"
                f" {self.time})")
