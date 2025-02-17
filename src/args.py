from dataclasses import dataclass
from typing import Tuple

from llamafactory import hparams
from transformers import HfArgumentParser


def split_arg(arg):
    if isinstance(arg, str):
        return [item.strip() for item in arg.split(",")]
    return arg


@dataclass
class DataArguments(hparams.DataArguments):
    """
    Data argument class.
    """
    prepare_dir: str = "prepare"
    history_strategy: str = "hop"
    history_length: int = 30
    entity: bool = False
    relation: bool = False
    map_strategy: str = "global"
    time: str = "global"
    partition: str = "all"

    def __post_init__(self):
        super().__post_init__()
        if self.partition == "all":
            self.partition = "train,valid,test"
        self.partition = split_arg(self.partition)

    def version_suffix(self) -> str:
        """
        Get version suffix of the data process method.
        """
        version = (
            f"{self.history_strategy}"
            f"-{self.history_length}"
            f"-{self.map_strategy}"
            f"-{self.time}"
        )
        if self.entity:
            version += "-ent"
        if self.relation:
            version += "-rel"
        if self.max_samples is not None:
            version += f"-{self.max_samples}"
        return version


@dataclass
class ModelArguments(hparams.ModelArguments):
    """
    Model argument class.
    """
    num_predictions: int = 30
    remove_duplicates: bool = True


@dataclass
class FinetuningArguments(hparams.FinetuningArguments):
    """
    Finetuning argument class.
    """
    checkpoint: int = None


_PREPARE_ARGS = [DataArguments]
_PREPARE_CLS = Tuple[DataArguments]
_EVAL_ARGS = [
    DataArguments,
    ModelArguments,
    FinetuningArguments,
]
_EVAL_CLS = Tuple[
    DataArguments,
    ModelArguments,
    FinetuningArguments,
]


def get_prepare_args(path: str) -> _PREPARE_CLS:
    """
    Get prepare phase arguments.
    """
    data_args, = HfArgumentParser(_PREPARE_ARGS).parse_yaml_file(path)
    return data_args,


def get_eval_args(path: str) -> _EVAL_CLS:
    """
    Get evaluation phase arguments.
    """
    data_args, model_args, finetuning_args = (
        HfArgumentParser(_EVAL_ARGS).parse_yaml_file(path))

    model_args.device_map = "auto"

    return data_args, model_args, finetuning_args
