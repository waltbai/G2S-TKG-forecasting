from abc import ABC, abstractmethod

from tqdm import tqdm
from typing import List

from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import TKG


class HistoryFinder(ABC):
    """Historical fact finder for queries."""

    def __init__(
            self,
            history_type: str = "entity",
            history_direction: str = "uni",
            history_length: int = 30,
            use_tqdm: bool = False,
    ):
        """Constructor of history finder.

        Args:
            history_type (str):
                Type of matching facts, either by "entity" or "pair".
                Defaults to "entity".
            history_direction (str):
                Direction of matching facts, either "uni" or "bi" direction.
                Defaults to "uni".
            history_length (int):
                Length of history facts.
                Defaults to 30.
            use_tqdm (bool):
                Whether to use tqdm or not.
                Defaults to False.
        """
        self.history_type = history_type
        self.history_direction = history_direction
        self.history_length = history_length
        self.use_tqdm = use_tqdm

    def __call__(
            self,
            queries: List[Query],
            tkg: TKG,
    ):
        with tqdm(total=len(queries), disable=not self.use_tqdm) as pbar:
            for query in queries:
                query.history = self.find_history(query, tkg)
                pbar.update()

    @abstractmethod
    def find_history(
            self,
            query: Query,
            tkg: TKG,
    ) -> List[Fact]:
        """Find history for query."""
