import json
import logging
import os

from datasets import Dataset
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
)

from llama_factory.llmtuner.hparams import FinetuningArguments
from llama_factory.llmtuner.model import load_model
from .args import PrepareArguments, TsroModelArguments, postprocess_args
from .inference import evaluate
from .prepare import get_data_name, prepare
from .train import train
from src.utils.metric import format_metrics

logger = logging.getLogger(__name__)


ARGS = (
    PrepareArguments,
    TsroModelArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
)


if __name__ == "__main__":
    # Parse arguments
    prepare_args, model_args, training_args, finetuning_args = \
        HfArgumentParser(ARGS).parse_args_into_dataclasses()
    postprocess_args(prepare_args, model_args, training_args, finetuning_args)

    # ===== Step 1: Prepare data =====
    prepare_path = os.path.join(
        prepare_args.prepare_dir,
        get_data_name(prepare_args)
    )
    # Check whether to call prepare() function
    prepare_flag = False
    if os.path.exists(prepare_path):
        if prepare_args.overwrite_prepare:
            logger.info("Data already exists. Overwrite existing ones.")
            prepare_flag = True
        else:
            logger.info("Data already exists. Skip prepare phase.")
    else:
        logger.info("Prepare data.")
        prepare_flag = True

    # Call prepare function
    if prepare_flag:
        prepare_parts = []
        if training_args.do_train:
            prepare_parts.append("train")
        if training_args.do_eval:
            prepare_parts.append("valid")
        if training_args.do_predict:
            prepare_parts.append("test")
        prepare(
            prepare_args,
            prepare_parts=prepare_parts,
            prepare_path=prepare_path,
        )

    # Load model and tokenizer
    logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        truncation_side="left",
        padding_side="left",
        model_max_length=1024,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_backbone = model_args.model_name_or_path.strip("/").split("/")[-1]
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
    )
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # ===== Step 2: Construct dataset =====
    logger.info("Load prepared data.")
    with open(prepare_path, "r") as f:
        prepare_data = json.load(f)
    train_set = Dataset.from_dict(prepare_data["train"])
    valid_set = Dataset.from_dict(prepare_data["valid"])
    test_set = Dataset.from_dict(prepare_data["test"])

    # ===== Step 3: Training =====
    if training_args.do_train:
        logger.info("Start training.")
        train(
            train_set=train_set,
            valid_set=valid_set,
            model=model,
            tokenizer=tokenizer,
            prepare_args=prepare_args,
            model_args=model_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
        )

    # ===== Step 4: Eval =====
    if training_args.do_eval:
        logger.info("Start evaluation.")
        metrics = evaluate(
            eval_set=valid_set,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")

    # ===== Step 5: Predict =====
    if training_args.do_predict:
        logger.info("Start testing.")
        metrics = evaluate(
            eval_set=test_set,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")
