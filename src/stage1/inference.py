import logging

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict

from src.args import ModelArguments
from src.utils.metric import compute_hits

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
                #   which may accidentally predict these IDs.
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
