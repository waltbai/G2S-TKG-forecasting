from dataclasses import dataclass


@dataclass
class BasePrepareArgument:
    """Basic PrepareArgument class."""
    dataset_dir: str
    dataset: str
    output_dir: str
    do_train: bool
    do_evaluate: bool
    do_predict: bool


@dataclass
class TsroPrepareArgument(BasePrepareArgument):
    """PrepareArgument class for TSRO prompts."""
    anonymize_entity: bool
    anonymize_rel: bool
    anonymize_time: bool
    history_length: int
    history_type: str
    history_direction: str
    num_predictions: int
