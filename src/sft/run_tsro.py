import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Literal, List, Dict, Callable

import ipdb
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    EvalPrediction,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, PreTrainedModel,
)

from llama_factory.llmtuner.hparams import FinetuningArguments
from llama_factory.llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from src.prepare.tsro import prepare
from src.utils.metric import compute_hits


@dataclass
class PrepareArguments:
    """Arguments for dataset preparation."""
    # Path settings
    dataset_dir: str = "data"
    dataset: Literal["ICEWS14", "ICEWS05-15", "ICEWS18", "WIKI", "YAGO"] = "ICEWS14"
    prepare_dir: str = "prepare"
    # Prompt settings
    anonymize_entity: bool = False
    anonymize_rel: bool = False
    anonymize_time: bool = True
    history_length: int = 30
    history_type: Literal["entity", "pair"] = "entity"
    history_direction: Literal["uni", "bi"] = "uni"


@dataclass
class ModelArguments:
    """Arguments for model."""
    model_name_or_path: str = "openai-community/gpt2"
    num_predictions: int = 30


def preprocess_func(
        tokenizer: PreTrainedTokenizer,
) -> Callable:
    """Wrap tokenize function to pass the tokenizer."""
    def tokenize_func(sample):
        """Tokenize function."""
        result = tokenizer(
            text=sample["prompt"],
            text_target=sample["answer"],
            truncation=True,
        )
        # Restrict labels to be 1 token
        result["labels"] = [_[:1] for _ in result["labels"]]
        return result

    return tokenize_func


def prepare_compute_metrics(
    tokenizer: PreTrainedTokenizer,
    answer: List[str],
    candidates: List[Dict[str, str]],
    filters: List[List[str]],
)->Callable:
    """Prepare compute_metrics method."""
    def compute_metrics(eval_pred: EvalPrediction):
        """Compute metrics."""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        num_samples = len(labels)
        num_predictions = len(predictions) // num_samples
        predictions[predictions < 0] = 50256
        predictions = predictions.reshape((num_samples, num_predictions, -1))
        results = [
            tokenizer.batch_decode(predictions[i], skip_special_tokens=True)
            for i in range(num_samples)
        ]
        preds = [
            [cand_dict[_] for _ in cand if _ in cand_dict and cand_dict[_] is not None]
            for cand, cand_dict in zip(results, candidates)
        ]
        labels = [cand_dict[a] for a, cand_dict in zip(answer, candidates)]
        return compute_hits(preds, labels, filters)

    return compute_metrics


def predict_1(
        test_set: Dataset,
        model: PreTrainedModel,
        num_predictions: int = 30,
):
    model.to("cuda:0")
    model.eval()
    tot_num = len(test_set)
    with torch.no_grad(), tqdm(total=tot_num) as pbar:
        tot_preds = []
        tot_answers = []
        tot_filters = []
        for sample in test_set:
            outputs = model.generate(
                sample["input_ids"].to("cuda:0").unsqueeze(0),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.unk_token_id,
            )
            _, probs_idx = torch.sort(outputs.scores[0], dim=-1, descending=True)
            probs_idx = probs_idx[0, :num_predictions].unsqueeze(1)
            results = tokenizer.batch_decode(probs_idx)
            candidates = sample["candidates"]
            preds = []
            for cand_id in results:
                cand_id = cand_id.strip()
                if cand_id in candidates and candidates[cand_id] is not None:
                    preds.append(candidates[cand_id])
            tot_preds.append(preds)
            tot_answers.append(candidates[sample["answer"]])
            tot_filters.append(sample["filters"])
            pbar.update()
    return compute_hits(
        tot_preds,
        tot_answers,
        tot_filters,
    )


def predict_2(
        test_set: Dataset,
        model: PreTrainedModel,
        num_predictions: int = 30,
):
    model.to("cuda:0")
    model.eval()
    tot_num = len(test_set)
    with torch.no_grad(), tqdm(total=tot_num) as pbar:
        tot_preds = []
        tot_answers = []
        tot_filters = []
        for sample in test_set:
            outputs = model.generate(
                sample["input_ids"].to("cuda:0").unsqueeze(0),
                max_new_tokens=1,
                pad_token_id=tokenizer.unk_token_id,
                num_beams=num_predictions,
                num_return_sequences=num_predictions,
            )
            outputs = outputs[:, -1:]
            results = tokenizer.batch_decode(outputs)
            candidates = sample["candidates"]
            preds = []
            for cand_id in results:
                cand_id = cand_id.strip()
                if cand_id in candidates and candidates[cand_id] is not None:
                    preds.append(candidates[cand_id])
            tot_preds.append(preds)
            tot_answers.append(candidates[sample["answer"]])
            tot_filters.append(sample["filters"])
            pbar.update()
    return compute_hits(
        tot_preds,
        tot_answers,
        tot_filters,
    )


if __name__ == '__main__':
    experiment_name = "experiment.tsro"
    logger = logging.getLogger(experiment_name)
    # Load arguments
    parser = HfArgumentParser(
        (
            PrepareArguments,
            ModelArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments,
        )
    )
    prepare_args, model_args, training_args, finetuning_args = \
        parser.parse_args_into_dataclasses()
    # Prepare
    prepare(
        dataset_dir=prepare_args.dataset_dir,
        dataset=prepare_args.dataset,
        prepare_dir=prepare_args.prepare_dir,
        anonymize_entity=prepare_args.anonymize_entity,
        anonymize_rel=prepare_args.anonymize_rel,
        anonymize_time=prepare_args.anonymize_time,
        history_length=prepare_args.history_length,
        history_type=prepare_args.history_type,
        history_direction=prepare_args.history_direction,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=training_args.do_predict,
        use_tqdm=True,
    )
    # Load dataset
    logger.info("Load datasets.")
    dataset_path = os.path.join(
        prepare_args.prepare_dir,
        f"{prepare_args.dataset}_TSRO.json"
    )
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    train_set = Dataset.from_dict(dataset["train"])
    valid_set = Dataset.from_dict(dataset["valid"])
    test_set = Dataset.from_dict(dataset["test"])
    # Tokenize
    logger.info("Tokenize")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        truncation_side="left",
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenized_set = test_set.map(
        preprocess_func(tokenizer),
        batched=True,
    ).with_format("torch")
    # Load model
    logger.info("Train and predict.")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path
    )
    model_backbone = model_args.model_name_or_path.split("/")[-1] + "-ICL"
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    metric_func = prepare_compute_metrics(
        tokenizer=tokenizer,
        answer=tokenized_set["answer"],
        candidates=tokenized_set["candidates"],
        filters=tokenized_set["filters"],
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_func,
    )
    # ipdb.set_trace(context=7)
    print(trainer.predict(
        test_dataset=tokenized_set,
        max_new_tokens=1,
        pad_token_id=tokenizer.unk_token_id,
        num_beams=model_args.num_predictions,
        num_return_sequences=model_args.num_predictions,
    ).metrics)
    # print(predict_1(tokenized_set, model, 30))
    # print(predict_2(tokenized_set, model, 30))
    # predict(
    #     test_set=tokenized_set,
    #     model=model,
    #     num_predictions=model_args.num_predictions,
    # )
