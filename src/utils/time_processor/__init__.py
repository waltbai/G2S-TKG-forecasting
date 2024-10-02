from src.utils.time_processor.base import TimeProcessor
from src.utils.time_processor.query import QueryTimeProcessor
from src.utils.time_processor.start import StartTimeProcessor


def get_time_processor(
        strategy: str,
        use_tqdm: bool = False,
) -> TimeProcessor:
    """Get time processor by strategy name."""
    if strategy == "start":
        return StartTimeProcessor(use_tqdm)
    else:   # strategy == "query"
        return QueryTimeProcessor(use_tqdm)
