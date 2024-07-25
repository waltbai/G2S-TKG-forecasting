from typing import Dict

from src.utils.query import Query
from src.utils.time_processor.base import TimeProcessor
from src.utils.tkg import TKG


class StartTimeProcessor(TimeProcessor):
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
