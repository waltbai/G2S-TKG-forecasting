from typing import List

from llm4tkg.preprocess.fact import Fact


def quadruple_prompt(
        query: Fact,
        history: List[Fact],
        anonymous: bool = False,
) -> str:
    """Construct quadruple-like prompt."""
    result = ""
    # Append historical facts
    for fact in history:
        # Append time
        if anonymous:
            result += f"{fact.time_idx}:[{fact.head_idx},{fact.rel_idx},{fact.tail_idx}]\n"
        else:
            result += f"{fact.time}:[{fact.head},{fact.rel},{fact.tail}]\n"
    # Append query
    if anonymous:
        result += f"{query.time_idx}:[{query.head_idx},{query.rel_idx},"
    else:
        result += f"{query.time}:[{query.head},{query.rel},"
    return result
