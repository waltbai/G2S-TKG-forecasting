from typing import Dict, Tuple, List, Any, Set

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm4tkg.preprocess.fact import Fact
from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.prompt import quadruple_prompt


def time_filter(
        predictions: List[Dict[str, Any]],
        query: Fact,
        time_filter_set: Set[Tuple[str, str, str, str]],
        query_target: str = "tail",
) -> List[Dict[str, Any]]:
    """Filter valid predictions."""
    results = []
    for item in predictions:
        if query_target == "tail":
            quad = (query.head, query.rel, item["prediction"], query.time)
            ground_truth = query.tail
        else:
            quad = (item["prediction"], query.rel, query.tail, query.time)
            ground_truth = query.head
        if item["prediction"] != ground_truth and quad in time_filter_set:
            continue
        results.append(item)
    return results


def metric(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute metrics."""
    hit1, hit3, hit10, total = 0, 0, 0, 0
    for result in results:
        for rank, pred in enumerate(result["predictions"]):
            if pred["entity"] == result["answer"]:
                if 0 <= rank < 1:
                    hit1 += 1
                if 0 <= rank < 3:
                    hit3 += 1
                if 0<= rank < 10:
                    hit10 += 1
            total += 1
    hit1 = hit1 / total
    hit3 = hit3 / total
    hit10 = hit10 / total
    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@10": hit10,
    }


class InContextLearningModel:
    """In-context learning model without supervised fine-tuning.

    This is a re-implementation of (Lee, et al., emnlp-2023).
    """

    def __init__(
            self,
            backbone: str = "openai-community/gpt2",
            device: str = "cpu",
            fp16: bool = False,
            history_length: int = 30,
            history_type: str = "entity",
            history_direction: str = "uni",
            top_k: int = 10,
            predict_head: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device=device,
        )
        self.device = device
        self.history_length = history_length
        self.history_type = history_type
        self.history_direction = history_direction
        self.top_k = top_k
        self.predict_head = predict_head



    def predict(
            self,
            prompt: str,
            candidates: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Predict a single sample."""
        model = self.model
        tokenizer = self.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
        )
        probs = outputs.scores[0]
        probs_idx = torch.argsort(probs, dim=-1, descending=True)
        preserve_idx = probs_idx[0][:self.top_k]
        results = []
        for tok_id in preserve_idx:
            ent_id = tokenizer.decode(tok_id).strip()
            if ent_id in candidates:
                results.append({
                    "entity": candidates[ent_id],
                    "score": probs[0][tok_id].item()
                })
        return results

    def evaluate(
            self,
            tkg: TemporalKG,
    ) -> None:
        """Evaluate on test set."""
        self.model.eval()
        facts = tkg.train_set + tkg.valid_set + tkg.test_set
        time_filter_set = set()
        for fact in tkg.test_set:
            time_filter_set.add(
                (fact.head, fact.rel, fact.tail, fact.time)
            )
        with torch.no_grad(), tqdm(total=len(tkg.test_set)) as pbar:
            results = []
            for query in tkg.test_set:
                # Construct prompt
                prompt, candidates = quadruple_prompt(
                    query=query,
                    facts=facts,
                    history_length=self.history_length,
                    history_type=self.history_type,
                    history_direction=self.history_direction,
                    anonymous=False,
                    anonymous_time=True,
                    shuffle=False,
                    query_target="tail",
                )
                # Generate results
                predictions = self.predict(
                    prompt=prompt,
                    candidates=candidates,
                )
                results.append({
                    "prompt": prompt,
                    "answer": query.tail,
                    "candidates": candidates,
                    "predictions": predictions,
                })
                # Check whether head entity should be predicted
                if self.predict_head:
                    prompt, candidates = quadruple_prompt(
                        query=query,
                        facts=facts,
                        history_length=self.history_length,
                        history_type=self.history_type,
                        history_direction=self.history_direction,
                        anonymous=False,
                        anonymous_time=True,
                        shuffle=False,
                        query_target="head",
                    )
                    predictions = self.predict(
                        prompt=prompt,
                        candidates=candidates,
                    )
                    results.append({
                        "prompt": prompt,
                        "answer": query.head,
                        "candidates": candidates,
                        "predictions": predictions,
                    })
                pbar.update()
        metrics = metric(results)
        print(metrics)
