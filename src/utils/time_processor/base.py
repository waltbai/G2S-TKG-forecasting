from abc import ABC, abstractmethod

from typing import List, Dict

from tqdm import tqdm

from src.utils.query import Query
from src.utils.tkg import TKG


class TimeProcessor(ABC):
    """Basic time processor class."""

    def __init__(self, use_tqdm: bool = False):
        self.use_tqdm = use_tqdm

    def __call__(
            self,
            queries: List[Query],
            tkg: TKG,
            **kwargs,
    ):
        with tqdm(total=len(queries), disable=not self.use_tqdm) as pbar:
            for query in queries:
                time_mapping = self.process(query=query, tkg=tkg, **kwargs)
                query.time_mapping = time_mapping
                pbar.update()

    @abstractmethod
    def process(
            self,
            query: Query,
            tkg: TKG,
            **kwargs,
    ) -> Dict[str, str]:
        """Process time of each query."""
        
