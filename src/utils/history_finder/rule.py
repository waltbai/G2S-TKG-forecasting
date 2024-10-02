from typing import List, Tuple

from src.utils.fact import Fact
from src.utils.history_finder.base import HistoryFinder
from src.utils.query import Query
from src.utils.tkg import TKG


class TemporalRuleHistoryFinder(HistoryFinder):
    """Find historical facts for queries via rules."""

    def find_history(
            self,
            query: Query,
            tkg: TKG,
            temporal_window=50,
    ) -> List[Fact]:
        """Find history for query."""
        rule_head = query.rel
        if rule_head not in tkg.rules:
            return []
        rule_bodies = tkg.rules[rule_head]
        sorted_bodies = sorted(rule_bodies.items(), key=lambda x: x[1], reverse=True)
        history = []
        for rule_body, confidence in sorted_bodies:
            key1 = f"{query.entity_role}_rel"
            key2 = (query.entity, rule_body)
            if key2 in tkg.search_history[key1]:
                temp_history = tkg.search_history[key1][key2]
            else:
                continue
            temp_history = [
                fact for fact in temp_history
                if fact.time < query.time
            ]
            if self.history_length:
                if len(history) + len(temp_history) < self.history_length:
                    history.extend(temp_history)
                else:
                    history.extend(temp_history[len(history)-self.history_length:])
                    break
            else:
                history.extend(temp_history)
        history = sorted(history, key=lambda x: x.time)
        return history
