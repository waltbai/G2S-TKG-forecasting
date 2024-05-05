import logging

import torch
from datasets import Dataset
from typing import Dict

from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from src.tsro.args import TsroModelArguments
from src.utils.metric import compute_hits

logger = logging.getLogger(__name__)


def evaluate(
        eval_set: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_args: TsroModelArguments,
        use_tqdm: bool = True,
) -> Dict[str, float]:
    """Evaluate on valid or test set."""
    model.eval()

    # Tokenize
    # Notice: this tokenize function is different from
    #   the one used in training, since we don't need
    #   to align input_ids with labels.
    logger.info("Process prepared data into evaluation format.")
    eval_set = eval_set.map(
        lambda x: tokenizer(x["prompt"], truncation=True),
        batched=True
    ).with_format("torch")

    # Inference
    logger.info("Start inference.")
    tot_preds = []
    tot_answers = []
    tot_filters = []
    tqdm_params = {"disable": not use_tqdm}
    with torch.no_grad(), tqdm(total=len(eval_set), **tqdm_params) as pbar:
        for sample in eval_set:
            # Get inputs
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
            answer = sample["answer"]
            filters = sample["filters"]
            candidates = sample["candidates"]

            # Generate
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
            # duplicate_set = set()
            for cand_id in results:
                cand_id = cand_id.strip()
                if cand_id in candidates and candidates[cand_id] is not None:
                    # if cand_id in duplicate_set and model_args.filter_duplicates:
                    #     continue
                    # duplicate_set.add(cand_id)
                    preds.append(candidates[cand_id])
            tot_preds.append(preds)
            tot_answers.append(candidates[answer])
            tot_filters.append(filters)

            # Update progress bar
            pbar.update()

    return compute_hits(tot_preds, tot_answers, tot_filters)
