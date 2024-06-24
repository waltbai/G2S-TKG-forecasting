import json
import logging
import os
import sys
from typing import Dict

import torch
import transformers
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM

from llamafactory.model import load_model
from src.args import get_infer_args, ModelArguments
from src.stage1.prepare import prepare, get_data_version
from src.utils.metric import compute_hits, format_metrics

logger = logging.getLogger(__name__)


def evaluate(
        eval_dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArguments,
        distributed_state: PartialState,
        use_tqdm: bool = True,
) -> Dict[str, float]:
    """Evaluate on valid or test dataset."""
    model.eval()

    # Tokenize func
    if distributed_state.is_main_process:
        logger.info("Process prepared data into evaluation format")
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x["prompt"], truncation=True),
        batched=True,
    ).with_format("torch")

    num_samples = len(eval_dataset)
    indices = list(range(num_samples))

    # Inference
    if distributed_state.is_main_process:
        logger.info("Start Evaluation")
    tot_preds, tot_answers, tot_filters = [], [], []
    cands = []
    with distributed_state.split_between_processes(
            indices,
            apply_padding=True,
    ) as parts:
        with torch.no_grad(), tqdm(
            total=len(parts),
            disable=not use_tqdm,
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
                candidates = sample["candidates"]

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

                # Decoder predictions
                preds = []
                duplicate_set = set()
                for cand_id in results:
                    cand_id = cand_id.strip()
                    # Dataset will automatically align candidate IDs,
                    #   model may accidentally predict these IDs.
                    #   Thus, we remove these automatically aligned IDs.
                    if cand_id in candidates and candidates[cand_id] is not None:
                        if cand_id in duplicate_set and model_args.remove_duplicates:
                            continue
                        duplicate_set.add(cand_id)
                        preds.append(candidates[cand_id])

                if len(preds) > 0:
                    cands.append(preds[0])
                else:
                    cands.append("")

                answer = candidates[label]
                tot_preds.extend([preds])
                tot_answers.extend([answer])
                tot_filters.extend([filters])
                pbar.update()

    tot_preds = gather_object(tot_preds)[:num_samples]
    tot_answers = gather_object(tot_answers)[:num_samples]
    tot_filters = gather_object(tot_filters)[:num_samples]

    cands = gather_object(cands)[:num_samples]
    # logger.info(f"Total counts: {len([_ for _ in cands if _ == '0'])}")

    return compute_hits(tot_preds, tot_answers, tot_filters)


if __name__ == "__main__":
    # Parse arguments from config file
    model_args, data_args, training_args, finetuning_args, generating_args = \
        get_infer_args(sys.argv[1], "stage1")

    # Prepare
    datafile_name = get_data_version(data_args) + ".json"
    data_path = os.path.join(data_args.prepare_dir, datafile_name)
    with open(data_path, "r") as f:
        dataset = json.load(f)
    valid_dataset = Dataset.from_dict(dataset["valid"])
    test_dataset = Dataset.from_dict(dataset["test"])

    # Initialize distributed inference
    distributed_state = PartialState()

    # Load model and backbone
    if distributed_state.is_main_process:
        logger.info(f"Load tokenizer and model from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        truncation_side="left",
        padding_side="left",
        model_max_length=data_args.cutoff_len,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_backbone = model_args.model_name_or_path.strip("/").split("/")[-1]
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=distributed_state.device,
    )
    # hack here: make model compatible with prediction
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    if training_args.do_eval:
        metrics = evaluate(
            eval_dataset=valid_dataset,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            distributed_state=distributed_state,
        )
        if distributed_state.is_main_process:
            logger.info(f"Results:\n{format_metrics(metrics)}")

    if training_args.do_predict:
        metrics = evaluate(
            eval_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            distributed_state=distributed_state,
        )
        if distributed_state.is_main_process:
            logger.info(f"Results:\n{format_metrics(metrics)}")

