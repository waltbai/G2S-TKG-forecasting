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

    def __str__(self):
        return (f"({self.head},"
                f" {self.rel},"
                f" {self.tail},"
                f" {self.time})")
