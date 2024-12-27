import json
import logging
import os
import sys

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from typing import Dict, Any, List

from llamafactory.model import load_tokenizer, load_model
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.stage2.args import get_eval_args, DataArguments, ModelArguments, FinetuningArguments
from src.utils.metric import format_metrics, compute_metrics

logger = logging.getLogger(__name__)


def evaluate(
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        distributed_state: PartialState,
        num_predictions: int = 30,
) -> Dict[str, float]:
    """
    Evaluate function.
    """
    # Tokenize dataset
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        logger.info("Process prepared data into evaluation format")
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x["input"], truncation=True),
        batched=True,
    ).with_format("torch")

    # Start evaluation
    distributed_state.wait_for_everyone()
    num_samples = len(eval_dataset)
    indices = list(range(num_samples))
    if distributed_state.is_main_process:
        logger.info("Start Evaluation")
    distributed_state.wait_for_everyone()
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
                label = sample["output"]
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
                probs_idx = probs_idx[0, :num_predictions].unsqueeze(1)
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


def convert_dataset(eval_dataset: List[Dict[str, Any]]) -> Dataset:
    """
    Convert dict dataset to Dataset format.
    """
    dataset = {
        "instruction": [],
        "input": [],
        "output": [],
        "filters": [],
        "id2entity": [],
    }
    for item in eval_dataset:
        dataset["instruction"].append(item["instruction"])
        dataset["input"].append(item["input"])
        dataset["output"].append(item["output"])
        dataset["filters"].append(item["filters"])
        dataset["id2entity"].append(item["id2entity"])
    return Dataset.from_dict(dataset)


def main():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    data_args, model_args, finetuning_args = get_eval_args(sys.argv[1])
    logger.addHandler(logging.FileHandler("eval.log"))

    distributed_state = PartialState()

    if finetuning_args.checkpoint is None:
        # Evaluate all checkpoints
        adapter_base_path = model_args.adapter_name_or_path[0]
        adapter_paths = []
        for d in os.listdir(adapter_base_path):
            if d.startswith("checkpoint-"):
                adapter_paths.append(os.path.join(adapter_base_path, d))
    else:
        # Evaluate certain checkpoint
        adapter_path = os.path.join(
            model_args.adapter_name_or_path[0],
            f"checkpoint-{finetuning_args.checkpoint}",
        )
        if not os.path.exists(adapter_path):
            raise ValueError(f"'{adapter_path}' not found!")
        adapter_paths = [adapter_path]
    adapter_paths = sorted(adapter_paths)

    # Load tokenizer and model
    for adapter_path in adapter_paths:
        distributed_state.wait_for_everyone()
        model_args.adapter_name_or_path = [adapter_path]
        if distributed_state.is_main_process:
            logger.info(f"===== Evaluating adapter {adapter_path} =====")
            logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
            logger.info(f"Load adapter from {model_args.adapter_name_or_path}")
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.model_max_length = data_args.cutoff_len
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

        for dataset in data_args.dataset:
            # Load prepared data
            version_suffix = data_args.version_suffix()
            valid_path = os.path.join(data_args.prepare_dir, f"{dataset}-valid-{version_suffix}.json")
            test_path = os.path.join(data_args.prepare_dir, f"{dataset}-test-{version_suffix}.json")
            with open(valid_path, "r") as f:
                valid_dataset = convert_dataset(json.load(f))
            with open(test_path, "r") as f:
                test_dataset = convert_dataset(json.load(f))

            # Valid
            metrics = evaluate(
                eval_dataset=valid_dataset,
                tokenizer=tokenizer,
                model=model,
                num_predictions=model_args.num_predictions,
                distributed_state=distributed_state,
            )
            output = format_metrics(metrics)
            distributed_state.wait_for_everyone()
            if distributed_state.is_main_process:
                logger.info(f"{dataset} Valid Set Results:\n{output}")

            # Test
            metrics = evaluate(
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                model=model,
                num_predictions=model_args.num_predictions,
                distributed_state=distributed_state,
            )
            output = format_metrics(metrics)
            distributed_state.wait_for_everyone()
            if distributed_state.is_main_process:
                logger.info(f"{dataset} Test Set Results:\n{output}")


if __name__ == "__main__":
    main()
