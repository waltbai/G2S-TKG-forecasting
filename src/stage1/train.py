import json
import logging
import os
import sys
from functools import partial

from datasets import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    DataCollatorForSeq2Seq,
)

from llamafactory.extras.callbacks import LogCallback
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.model.loader import TokenizerModule
from llamafactory.train.sft.metric import ComputeMetrics
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from src.args import get_train_args, AnonymizedDataArguments, ModelArguments, TrainingArguments, FinetuningArguments
from src.stage1.prepare import get_data_version

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
        model: PreTrainedModel,
        tokenizer_module: TokenizerModule,
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
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
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
    model_args, data_args, training_args, finetuning_args, generating_args = \
        get_train_args(sys.argv[1], "stage1")

    # Load prepared data
    datafile_name = get_data_version(data_args) + ".json"
    data_path = os.path.join(data_args.prepare_dir, datafile_name)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    train_dataset = Dataset.from_dict(dataset["train"])

    # Load model and backbone
    logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.model_max_length = data_args.cutoff_len
    model_backbone = model_args.model_name_or_path.strip("/").split("/")[-1]
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
    )

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = \
        training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = \
        data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = \
        False if model_args.visual_inputs else training_args.remove_unused_columns

    # Train train_dataset
    if training_args.do_train:
        train(
            train_dataset=train_dataset,
            model=model,
            tokenizer_module=tokenizer_module,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
        )
