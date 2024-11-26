import json
import logging
import os
import sys

import torch
from accelerate import PartialState, accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from typing import Dict

from llamafactory.model import load_tokenizer, load_model
from tqdm import tqdm

from src.stage2.args import (
    get_train_args,
    DataArguments,
    ModelArguments,
    FinetuningArguments,
    TrainingArguments,
)
from src.utils.metric import format_metrics, compute_metrics

logger = logging.getLogger(__name__)


def evaluate(
        eval_dataset: Dataset,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        finetuning_args: FinetuningArguments,
) -> Dict[str, float]:
    """
    Evaluate function.
    """
    distributed_state = PartialState()

    # Load model and tokenizer
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.model_max_length = data_args.cutoff_len
    accelerator.wait_for_everyone()
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Tokenize dataset
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        logger.info("Process prepared data into evaluation format")
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x["prompt"], truncation=True),
        batched=True,
    ).with_format("torch")

    # Start evaluation
    distributed_state.wait_for_everyone()
    num_samples = len(eval_dataset)
    indices = list(range(num_samples))
    if distributed_state.is_main_process:
        logger.info("Start Evaluation")
    tot_preds, tot_answers, tot_filters = [], [], []
    with distributed_state.split_between_processes(
            indices,
            apply_padding=True,
    ) as parts:
        with torch.no_grad(), tqdm(
                total=len(parts),
        ) as pbar:
            for sample_id in parts:
                sample = eval_dataset[sample_id]
                # Get inputs
                input_ids = sample["input_ids"].unsqueeze(0).to(
                    distributed_state.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(
                    distributed_state.device)
                label = sample["label"]
                filters = sample["filters"]
                id2entity = sample["id2entity"]

                # Generate output
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                _, probs_idx = torch.sort(outputs.scores[0], dim=-1, descending=True)
                probs_idx = probs_idx[0, :model_args.num_predictions].unsqueeze(1)
                results = tokenizer.batch_decode(probs_idx)

                # Decode predictions
                preds = []
                duplicate_set = set()
                for ent_id in results:
                    ent_id = ent_id.strip()
                    if ent_id in duplicate_set:
                        continue
                    if ent_id in id2entity and id2entity[ent_id] is not None:
                        duplicate_set.add(ent_id)
                        preds.append(id2entity[ent_id])

                tot_preds.append(preds)
                if label != "None":
                    tot_answers.append(id2entity[label])
                else:
                    tot_answers.append(None)
                tot_filters.append(filters)
                pbar.update()

    tot_preds = gather_object(tot_preds)[:num_samples]
    tot_answers = gather_object(tot_answers)[:num_samples]
    tot_filters = gather_object(tot_filters)[:num_samples]

    return compute_metrics(tot_preds, tot_answers, tot_filters)


if __name__ == "__main__":
    data_args, model_args, training_args, finetuning_args, generation_args = (
        get_train_args(sys.argv[1]))
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load prepared data
    datafile_name = data_args.get_data_version() + ".json"
    data_path = os.path.join(data_args.prepare_dir, datafile_name)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    valid_dataset = Dataset.from_dict(dataset["valid"])
    test_dataset = Dataset.from_dict(dataset["test"])

    # Valid
    model_args.adapter_name_or_path = [training_args.output_dir]
    training_args.do_train = False
    metrics = evaluate(
        eval_dataset=valid_dataset,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
    )
    output = format_metrics(metrics)
    logger.info(f"Results:\n{output}")

    # Test
    metrics = evaluate(
        eval_dataset=test_dataset,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
    )
    output = format_metrics(metrics)
    logger.info(f"Results:\n{output}")
