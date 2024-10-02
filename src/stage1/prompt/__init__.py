from src.stage1.prompt.base import PromptConstructStrategy
from src.stage1.prompt.qa import QAPromptConstructStrategy


def get_prompt_constructor(
        strategy: str,
        use_tqdm: bool = False,
) -> PromptConstructStrategy:
    if strategy == "qa":
        return QAPromptConstructStrategy(use_tqdm=use_tqdm)
    else:
        raise ValueError(f"Prompt strategy '{strategy}' is not supported")
