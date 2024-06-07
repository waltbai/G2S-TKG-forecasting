import logging
import os
from dataclasses import dataclass

import torch
import transformers
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import llama_factory.llmtuner.hparams as hp
from llama_factory.llmtuner.extras.constants import TRAINER_CONFIG
from llama_factory.llmtuner.extras.misc import get_current_device

logger = logging.getLogger(__name__)


@dataclass
class AnonymizedDataArguments(hp.DataArguments):
    """Data preparation arguments in stage-1."""
    prepare_dir: str = "prepare"
    train_dataset: str = ""
    valid_dataset: str = ""
    test_dataset: str = ""
    history_type: str = "entity"
    history_direction: str = "uni"
    history_length: int = 30
    anonymize_strategy: str = "global"
    anonymize_prefix: bool = False
    time_process_strategy: str = "start"
    vague_time: bool = False
    prompt_construct_strategy: str = "inline"


@dataclass
class DeAnonymizedDataArguments(hp.DataArguments):
    """Data preparation arguments in stage-2."""
    prepare_dir: str = "prepare"
    history_type: str = "entity"
    history_direction: str = "uni"
    history_length: int = 30
    anonymize_strategy: str = "global"
    anonymize_prefix: bool = False
    time_process_strategy: str = "start"
    vague_time: bool = False
    deanonymize_strategy: str = "fillin"
    prompt_construct_strategy: str = "inline"


@dataclass
class ModelArguments(hp.ModelArguments):
    """Model arguments class."""
    num_predictions: int = 30
    remove_duplicates: bool = False


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """Training arguments class."""


@dataclass
class GenerationArguments(hp.GeneratingArguments):
    """Generation arguments class."""


@dataclass
class FinetuningArguments(hp.FinetuningArguments):
    """Fine-tuning arguments class."""


def post_process_args(
        data_args: hp.DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        finetuning_args: FinetuningArguments,
        generation_args: GenerationArguments,
):
    """Post-process arguments."""
    if (
            training_args.parallel_mode.value == "distributed"
            and training_args.ddp_find_unused_parameters is None
            and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
            training_args.resume_from_checkpoint is None
            and training_args.do_train
            and os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
            and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        files = os.listdir(training_args.output_dir)
        if last_checkpoint is None and len(files) > 0 and (len(files) != 1 or files[0] != TRAINER_CONFIG):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                    training_args.resume_from_checkpoint
                )
            )

    if (
            finetuning_args.stage in ["rm", "ppo"]
            and finetuning_args.finetuning_type == "lora"
            and training_args.resume_from_checkpoint is not None
    ):
        logger.warning(
            "Add {} to `adapter_name_or_path` to resume training from checkpoint.".format(
                training_args.resume_from_checkpoint
            )
        )

        # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary:
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode.value == "distributed",
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns
