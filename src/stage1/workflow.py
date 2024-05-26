import json
import logging
import os
import sys

from datasets import Dataset
from transformers import HfArgumentParser, AutoTokenizer

from llama_factory.llmtuner.model import load_model
from src.args import (
    AnonymizedDataArguments,
    ModelArguments,
    TrainingArguments,
    FinetuningArguments,
    GenerationArguments,
    post_process_args,
)
from src.stage1.inference import evaluate
from src.stage1.prepare import prepare
from src.stage1.train import train
from src.utils.metric import format_metrics

logger = logging.getLogger(__name__)


def main():
    """Main controller function of stage-1."""
    # Parse arguments from config file
    config_path = sys.argv[1]
    parser = HfArgumentParser([
        AnonymizedDataArguments,
        ModelArguments,
        TrainingArguments,
        FinetuningArguments,
        GenerationArguments,
    ])
    data_args, model_args, training_args, finetuning_args, generation_args = \
        parser.parse_yaml_file(os.path.abspath(config_path))
    post_process_args(
        data_args,
        model_args,
        training_args,
        finetuning_args,
        generation_args,
    )

    # Prepare
    data_path = prepare(data_args)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    train_dataset = Dataset.from_dict(dataset["train"])
    valid_dataset = Dataset.from_dict(dataset["valid"])
    test_dataset = Dataset.from_dict(dataset["test"])

    # Load model and backbone
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
    # hack here: make model compatible with prediction
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Train train_dataset
    if training_args.do_train:
        train(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            model=model,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
        )

    # Evaluate valid_dataset
    if training_args.do_eval:
        metrics = evaluate(
            eval_dataset=valid_dataset,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")

    # Evaluate test_dataset
    if training_args.do_predict:
        metrics = evaluate(
            eval_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")


if __name__ == "__main__":
    main()
