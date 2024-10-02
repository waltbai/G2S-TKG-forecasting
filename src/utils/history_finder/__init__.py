from src.utils.history_finder.base import HistoryFinder
from src.utils.history_finder.one_hop import OneHopHistoryFinder
from src.utils.history_finder.rule import TemporalRuleHistoryFinder


def get_history_finder(
        history_finder: str = "1-hop",
        history_type: str = "entity",
        history_direction: str = "uni",
        history_length: int = 30,
        use_tqdm: bool = False,
) -> HistoryFinder:
    """Get history finder."""
    if history_finder == "1-hop":
        return OneHopHistoryFinder(
            history_type=history_type,
            history_direction=history_direction,
            history_length=history_length,
            use_tqdm=use_tqdm,
        )
    else:
        return TemporalRuleHistoryFinder(
            history_type=history_type,
            history_direction=history_direction,
            history_length=history_length,
            use_tqdm=use_tqdm,
        )
