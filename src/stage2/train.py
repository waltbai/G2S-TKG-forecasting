import json
import logging
import os
import sys
from functools import partial

from accelerate import accelerator
from datasets import Dataset
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_tokenizer, load_model
from llamafactory.model.loader import TokenizerModule
from llamafactory.train.callbacks import LogCallback
from llamafactory.train.sft.metric import ComputeSimilarity, ComputeAccuracy, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorForSeq2Seq

from src.stage2.args import get_train_args, DataArguments, ModelArguments, TrainingArguments, FinetuningArguments

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
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        finetuning_args: FinetuningArguments,
):
    """
    Train function.
    """
    # Load model and tokenizer
    logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.model_max_length = data_args.cutoff_len
    accelerator.wait_for_everyone()
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Tokenize and align inputs with labels
    logger.info("Process prepared data into training format.")
    tokenized_train_set = train_dataset.map(
        partial(preprocess_func, tokenizer=tokenizer),
        batched=True,
        remove_columns=["prompt", "label", "filters", "id2entity"]
    ).with_format("torch")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" \
            else None,  # for shift short attention
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss \
            else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    callbacks = [LogCallback(training_args.output_dir)]
    trainer = CustomSeq2SeqTrainer(
        model=model,
        train_dataset=tokenized_train_set,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
        **metric_module,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss"])


if __name__ == "__main__":
    data_args, model_args, training_args, finetuning_args, generation_args = (
        get_train_args(sys.argv[1]))

    # Load prepared data
    datafile_name = data_args.get_data_version() + ".json"
    data_path = os.path.join(data_args.prepare_dir, datafile_name)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    train_dataset = Dataset.from_dict(dataset)

    # Training
    training_args.do_train = True
    train(
        train_dataset=train_dataset,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
    )
