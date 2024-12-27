from dataclasses import dataclass

from llamafactory import hparams
from llamafactory.extras.logging import get_logger


logger = get_logger(__name__)


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
    map_strategy: str = "session"
    time: str = "session"

    def version_suffix(self) -> str:
        """
        Get version suffix of the data process method.
        """
        version = (
            "stage1"
            f"-{self.history_strategy}"
            f"-{self.history_length}"
            f"-{self.map_strategy}"
            f"-{self.time}"
        )
        if self.entity:
            version += f"-ent"
        if self.relation:
            version += "-rel"
        if self.max_samples is not None:
            version += f"-{self.max_samples}"
        return version
