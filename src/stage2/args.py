import os
from dataclasses import dataclass
from typing import Tuple

import torch
import transformers
from llamafactory import hparams
from llamafactory.extras.constants import CHECKPOINT_NAMES
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import get_current_device
from llamafactory.hparams.parser import _verify_model_args, _check_extra_dependencies, _set_transformers_logging
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, is_torch_npu_available
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available


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
    map_strategy: str = "global"
    time: str = "global"

    def get_data_version(self) -> str:
        """
        Get version of the data process method.
        """
        dataset = "_".join(self.dataset)
        version = (
            "stage2-"
            f"{dataset}"
            f"-{self.history_strategy}"
            f"-{self.history_length}"
            f"-{self.map_strategy}"
            f"-{self.time}"
        )
        if self.entity:
            version += "-ent"
        if self.relation:
            version += "-rel"
        return version


@dataclass
class ModelArguments(hparams.ModelArguments):
    """
    Model argument class.
    """
    num_predictions: int = 30
    remove_duplicates: bool = True


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
    Training argument class.
    """


@dataclass
class GenerationArguments(hparams.GeneratingArguments):
    """
    Generation argument class.
    """


@dataclass
class FinetuningArguments(hparams.FinetuningArguments):
    """
    Finetuning argument class.
    """


_PREPARE_ARGS = [DataArguments]
_PREPARE_CLS = Tuple[DataArguments]
_TRAIN_ARGS = [
    DataArguments,
    ModelArguments,
    TrainingArguments,
    FinetuningArguments,
    GenerationArguments,
]
_TRAIN_CLS = Tuple[
    DataArguments,
    ModelArguments,
    TrainingArguments,
    FinetuningArguments,
    GenerationArguments,
]


def get_prepare_args(path: str) -> _PREPARE_CLS:
    """
    Get prepare phase arguments.
    """
    data_args, = HfArgumentParser(_PREPARE_ARGS).parse_yaml_file(path)
    return data_args,


def get_train_args(path: str) -> _TRAIN_CLS:
    """
    Get train phase arguments.
    """
    data_args, model_args, training_args, finetuning_args, generation_args = (
        HfArgumentParser(_TRAIN_ARGS).parse_yaml_file(path))

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")

        if training_args.report_to and training_args.report_to[0] not in ["wandb", "tensorboard"]:
            raise ValueError("PPO only accepts wandb or tensorboard logger.")

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict) and (
            data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please specify dataset for evaluation.")

    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")

        if data_args.eval_dataset is None:
            raise ValueError("Cannot use `predict_with_generate` if `eval_dataset` is None.")

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")

    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if (
            finetuning_args.use_galore
            and finetuning_args.galore_layerwise
            and training_args.parallel_mode == ParallelMode.DISTRIBUTED
    ):
        raise ValueError("Distributed training does not support layer-wise GaLore.")

    if finetuning_args.use_badam and training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.badam_mode == "ratio":
            raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
        elif not is_deepspeed_zero3_enabled():
            raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")

    if finetuning_args.use_galore and training_args.deepspeed is not None:
        raise ValueError("GaLore is incompatible with DeepSpeed yet.")

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if data_args.neat_packing and not data_args.packing:
        logger.warning("`neat_packing` requires `packing` is True. Change `packing` to True.")
        data_args.packing = True

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if (
            training_args.do_train
            and finetuning_args.finetuning_type == "lora"
            and model_args.quantization_bit is None
            and model_args.resize_vocab
            and finetuning_args.additional_target is None
    ):
        logger.warning("Remember to add embedding layers to `additional_target` to make the added tokens trainable.")

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if training_args.do_train and finetuning_args.use_galore and not finetuning_args.pure_bf16:
        logger.warning("Using GaLore with mixed precision training may significantly increases GPU memory usage.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    if (
            training_args.parallel_mode == ParallelMode.DISTRIBUTED
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
        if last_checkpoint is None and any(
                os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info("Resuming training from {}.".format(training_args.resume_from_checkpoint))
            logger.info("Change `output_dir` or use `overwrite_output_dir` to avoid.")

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
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode == ParallelMode.DISTRIBUTED,
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return data_args, model_args, training_args, finetuning_args, generation_args
