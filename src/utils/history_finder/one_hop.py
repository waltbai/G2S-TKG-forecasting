from typing import List

from src.utils.fact import Fact
from src.utils.history_finder.base import HistoryFinder
from src.utils.query import Query
from src.utils.tkg import TKG


class OneHopHistoryFinder(HistoryFinder):
    """Find one hop historical facts for queries."""

    def find_history(
            self,
            query: Query,
            tkg: TKG,
    ) -> List[Fact]:
        """Find history for query."""
        if self.history_direction == "uni":
            if self.history_type == "entity":
                search_key_1 = query.entity_role
                search_key_2 = query.entity
            else:  # history_type == "pair"
                search_key_1 = f"{query.entity_role}_rel"
                search_key_2 = (query.entity, query.rel)
        else:  # history_direction == "bi"
            if self.history_type == "entity":
                search_key_1 = "both"
                search_key_2 = query.entity
            else:  # history_type == "pair"
                search_key_1 = "both_rel"
                search_key_2 = (query.entity, query.rel)
        facts = tkg.search_history[search_key_1][search_key_2]

        # Filter future facts and cut-off
        history = [fact for fact in facts if fact.time < query.time]
        history = history[-self.history_length:]
        return history
