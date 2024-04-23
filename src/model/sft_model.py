import logging
from typing import Any

from src.model.icl_model import InContextLearningModel
from src.preprocess.tkg import TemporalKG


class SupervisedFineTuneModel(InContextLearningModel):
    """Model with supervised fine-tuning."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.logging = logging.getLogger("SFTModel")

    def train(
            self,
            tkg: TemporalKG,
    ):
        """Train on train set."""
        
