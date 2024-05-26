from abc import ABC, abstractmethod

from typing import List, Dict

from tqdm import tqdm

from src.utils.query import Query
from src.utils.tkg import TKG


class TimeProcessStrategy(ABC):
    """Basic time process strategy class."""

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
        """Process each query."""


class AbsoluteTimeProcessStrategy(TimeProcessStrategy):
    """Use original absolute time."""

    def process(
            self,
            query: Query,
            tkg: TKG,
            **kwargs,
    ) -> Dict[str, str]:
        time_mapping = {query.time: query.time}
        for fact in query.history:
            time_mapping.setdefault(fact.time, fact.time)
        return time_mapping


class AbsoluteStartTimeProcessStrategy(TimeProcessStrategy):
    """Time are calculated from the start time."""

    def process(
            self,
            query: Query,
            tkg: TKG,
            **kwargs,
    ) -> Dict[str, str]:
        map_func = lambda x: str(tkg.time2id[x])
        time_mapping = {query.time: map_func(query.time)}
        for fact in query.history:
            time_mapping.setdefault(fact.time, map_func(fact.time))
        return time_mapping


class RelativeQueryTimeProcessStrategy(TimeProcessStrategy):
    """Time are calculated from the query time. """

    def process(
            self,
            query: Query,
            tkg: TKG,
            vague: bool = False,
            **kwargs,
    ) -> Dict[str, str]:
        map_func = lambda x: str(tkg.time2id[query.time] - tkg.time2id[x])
        time_mapping = {query.time: map_func(query.time)}
        for fact in query.history:
            time_mapping.setdefault(fact.time, map_func(fact.time))
        return time_mapping


def get_time_processor(
        strategy: str,
        use_tqdm: bool = False,
) -> TimeProcessStrategy:
    """Get time processor by strategy name."""
    if strategy == "absolute":
        return AbsoluteTimeProcessStrategy(use_tqdm)
    elif strategy == "start":
        return AbsoluteStartTimeProcessStrategy(use_tqdm)
    else:   # strategy == "query"
        return RelativeQueryTimeProcessStrategy(use_tqdm)
