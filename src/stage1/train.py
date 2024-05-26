import logging
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Union, Tuple, Dict

import numpy as np
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorForSeq2Seq

from llama_factory.llmtuner.extras.callbacks import LogCallback
from llama_factory.llmtuner.extras.constants import IGNORE_INDEX
from llama_factory.llmtuner.extras.ploting import plot_loss
from llama_factory.llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from src.args import AnonymizedDataArguments, ModelArguments, TrainingArguments, FinetuningArguments

logger = logging.getLogger(__name__)


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.

    Notice: modified since we do not need to handle Chinese word segmentation currently.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            # hypothesis = list(jieba.cut(pred))
            # reference = list(jieba.cut(label))
            hypothesis = pred.split(" ")
            reference = label.split(" ")

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


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
    tokenized_valid_set = valid_dataset.map(
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
        eval_dataset=tokenized_valid_set,
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
