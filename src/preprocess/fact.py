from dataclasses import dataclass

from typing import Tuple, List

from src.utils.common import card2ord


@dataclass
class Fact:
    """Universal class for temporal fact."""
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
            _format: str = "normal"
    ) -> Tuple[str, str, str, str]:
        """Quadruple representation.

        Args:
            _format (str): select from ["normal", "inverse", "swap"]
        """
        if _format == "normal":
            return (
                self.head,
                self.rel,
                self.tail,
                self.time,
            )
        elif _format == "inverse":
            return (
                self.tail,
                f"inverse {self.rel}",
                self.head,
                self.time,
            )
        else:
            return (
                self.tail,
                self.rel,
                self.head,
                self.time,
            )

    def quadruple_idx(
            self,
    ) -> Tuple[int, int, int, int]:
        """Quadruple index representation."""
        return (
            self.head_idx,
            self.rel_idx,
            self.tail_idx,
            self.time_idx,
        )

    def prompt_quadruple(
            self,
            query_entity: str,
            anonymize: bool = False,
            anonymize_time: bool = True,
    ) -> Tuple[str, str, str, str]:
        """Representation for quadruple prompt."""
        if anonymize:
            head, rel, tail = self.head_idx, self.rel_idx, self.tail_idx
        else:
            head, rel, tail = self.head, self.rel, self.tail
        if anonymize_time:
            time = self.time_idx
        else:
            time = self.time
        if query_entity == self.tail:
            head, tail = tail, head
        head, rel, tail, time = map(str, (head, rel, tail, time))
        return head, rel, tail, time

    def prompt_text(
            self,
            query_entity: str,
            anonymize: bool = False,
            anonymize_time: bool = True,
    ) -> Tuple[str, str, str, str]:
        """Representation for text prompt."""
        if anonymize:
            head, rel, tail = self.head_idx, self.rel_idx, self.tail_idx
        else:
            head, rel, tail = self.head, self.rel, self.tail
        if anonymize_time:
            time = card2ord(self.time_idx)
        else:
            time = self.time
        if query_entity == self.tail:
            head, tail = tail, head
        head, rel, tail, time = map(str, (head, rel, tail, time))
        # head = head.capitalize()
        # rel = rel.lower()
        # tail = tail.capitalize()
        return head, rel, tail, time


    def __str__(self):
        return (f"({self.head},"
                f" {self.rel},"
                f" {self.tail},"
                f" {self.time})")
