import json
import logging
import os
import sys
from typing import Dict

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, HfArgumentParser, AutoTokenizer, AutoModelForCausalLM

from llama_factory.llmtuner.model import load_model
from src.args import ModelArguments, AnonymizedDataArguments, TrainingArguments, FinetuningArguments, \
    GenerationArguments, post_process_args
from src.stage1.prepare import prepare
from src.utils.metric import compute_hits, format_metrics

logger = logging.getLogger(__name__)


def evaluate(
        eval_dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArguments,
        use_tqdm: bool = True,
) -> Dict[str, float]:
    """Evaluate on valid or test dataset."""
    model.eval()

    # Tokenize func
    logger.info("Process prepared data into evaluation format")
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x["prompt"], truncation=True),
        batched=True
    ).with_format("torch")

    # Inference
    logger.info("Start Evaluation")
    tot_preds, tot_answers, tot_filters = [], [], []
    with torch.no_grad(), tqdm(total=len(eval_dataset), disable=not use_tqdm) as pbar:
        for sample in eval_dataset:
            # Get inputs
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
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
            answer = candidates[label]
            tot_preds.append(preds)
            tot_answers.append(answer)
            tot_filters.append(filters)
            pbar.update()

    return compute_hits(tot_preds, tot_answers, tot_filters)


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
    # Change model path
    model_args.model_name_or_path = training_args.output_dir

    # Prepare
    data_path = prepare(data_args)
    with open(data_path, "r") as f:
        dataset = json.load(f)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
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
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")

    if training_args.do_predict:
        metrics = evaluate(
            eval_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
        )
        logger.info(f"Results:\n{format_metrics(metrics)}")

