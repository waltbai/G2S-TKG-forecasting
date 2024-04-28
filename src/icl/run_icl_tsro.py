import argparse
import json
import logging
import os

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any

from src.utils.common import format_params
from src.utils.metric import compute_metrics, format_metrics


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str, default="/data/bailong/models/gpt2"
    )
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--dataset", default="ICEWS14", type=str,
                        choices=["ICEWS14", "ICEWS05-15", "ICEWS18", "WIKI", "YAGO"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_predictions", type=int, default=30)
    return parser.parse_args()

def predict(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_set: Dataset,
        device: str = "cuda:0",
        num_predictions: int = 30,
        filter_duplicate: bool = False,
        use_tqdm: bool = True,
) -> Dict[str, float]:
    """Predict on test set."""
    model.to(device)
    model.eval()
    tot_num = len(test_set)
    tqdm_params = {"ascii": False, "disable": not use_tqdm}
    with torch.no_grad(), tqdm(total=tot_num, **tqdm_params) as pbar:
        tot_preds = []
        tot_answers = []
        tot_filters = []
        for sample in test_set:
            inputs = {k: v.to(device) for k, v in sample["inputs"].items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            _, probs_idx = torch.sort(outputs.scores[0], dim=-1, descending=True)
            probs_idx = probs_idx[0, :num_predictions]
            # Decode to token
            answer = sample["answer"]
            filters = sample["filters"]
            candidates = sample["candidates"]
            preds = []
            pred_ids = set()
            for token_id in probs_idx:
                cand_id = tokenizer.decode(token_id).strip()
                # Dataset class will automatically
                # align key of candidates across different samples.
                if cand_id in candidates and candidates[cand_id] is not None:
                    if filter_duplicate and cand_id in pred_ids:
                        continue
                    pred_ids.add(cand_id)
                    entity = candidates[cand_id]
                    preds.append(entity)
            tot_preds.append(preds)
            tot_answers.append(answer)
            tot_filters.append(filters)
            pbar.update()
    return compute_metrics(tot_preds, tot_answers, tot_filters)


if __name__ == '__main__':
    logger = logging.getLogger("ICLmodel")
    args = get_args()
    # Load data
    logger.info("Load test data.")
    dataset_path = os.path.join(args.dataset_dir, f"{args.dataset}_TSRO.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    test_set = Dataset.from_dict(dataset["test"])
    # Tokenize
    logger.info("Tokenize.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        truncation_side="left"
    )
    tokenized_set = test_set.map(
        lambda x: {
            "inputs": tokenizer(x["prompt"], return_tensors="pt", truncation=True)
        }
    ).with_format("torch")
    # Load model
    logger.info("Predict on test set.")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model_backbone = args.model_name_or_path.split("/")[-1] + "-ICL"
    # Inference
    metrics = predict(
        model=model,
        tokenizer=tokenizer,
        test_set=tokenized_set,
        device=args.device,
        num_predictions=args.num_predictions,
    )
    # Log results
    exp_settings = [
        ("dataset", args.dataset),
        ("model", model_backbone),
        ("num_predictions", args.num_predictions),
    ]
    logger.info(f"Experiment settings:\n{format_params(exp_settings)}")
    logger.info(f"Experiment results:\n{format_metrics(metrics)}")
