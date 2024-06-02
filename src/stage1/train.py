import json
import logging
import os
import sys
from functools import partial

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorForSeq2Seq, HfArgumentParser, AutoTokenizer

from llama_factory.llmtuner.extras.callbacks import LogCallback
from llama_factory.llmtuner.extras.ploting import plot_loss
from llama_factory.llmtuner.model import load_model
from llama_factory.llmtuner.train.sft.metric import ComputeMetrics
from llama_factory.llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from src.args import AnonymizedDataArguments, ModelArguments, TrainingArguments, FinetuningArguments, \
    GenerationArguments, post_process_args
from src.stage1.prepare import prepare

logger = logging.getLogger(__name__)


def preprocess_func(
        sample,
        tokenizer: PreTrainedTokenizer,
):
    """Preprocess prompts and answers."""
    texts = [
        prompt + answer
        for prompt, answer in zip(sample["prompt"], sample["label"])
    ]
    results = tokenizer(texts, truncation=True)
    aligned_results = {"input_ids": [], "attention_mask": [], "labels": []}
    for input_ids, attention_mask in \
            zip(results["input_ids"], results["attention_mask"]):
        labels = [-100] * len(input_ids)
        labels[-1] = input_ids[-1]
        aligned_results["input_ids"].append(input_ids)
        aligned_results["attention_mask"].append(attention_mask)
        aligned_results["labels"].append(labels)
    return aligned_results


def train(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data_args: AnonymizedDataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        finetuning_args: FinetuningArguments,
):
    """Train on train and valid set."""
    # Tokenize and align inputs with labels
    logger.info("Process prepared data into training format.")
    tokenized_train_set = train_dataset.map(
        partial(preprocess_func, tokenizer=tokenizer),
        batched=True,
        remove_columns=["prompt", "label", "filters", "candidates"]
    ).with_format("torch")
    # tokenized_valid_set = valid_dataset.map(
    #     partial(preprocess_func, tokenizer=tokenizer),
    #     batched=True,
    #     remove_columns=["prompt", "label", "filters", "candidates"]
    # ).with_format("torch")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" \
            else None,  # for shift short attention
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss \
            else tokenizer.pad_token_id,
    )

    callbacks = [LogCallback(training_args.output_dir)]

    trainer = CustomSeq2SeqTrainer(
        model=model,
        train_dataset=tokenized_train_set,
        # eval_dataset=tokenized_valid_set,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


if __name__ == "__main__":
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
    

