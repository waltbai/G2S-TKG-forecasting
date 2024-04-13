from typing import Dict, Tuple, List, Any, Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm4tkg.evaluation import metric, QueryResult, Prediction
from llm4tkg.preprocess.fact import Fact
from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.prompt import quadruple_prompt


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
            top_k: int = 30,
            predict_head: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            truncation_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )
        self.model.to(device)
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
    ) -> List[Prediction]:
        """Predict a single sample."""
        model = self.model
        tokenizer = self.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
            pad_token_id = tokenizer.eos_token_id
        )
        probs = outputs.scores[0]
        probs_idx = torch.argsort(probs, dim=-1, descending=True)
        preserve_idx = probs_idx[0][:self.top_k]
        results = []
        for tok_id in preserve_idx:
            ent_id = tokenizer.decode(tok_id).strip()
            if ent_id in candidates:
                results.append(Prediction(
                    entity=candidates[ent_id],
                    score=probs[0][tok_id].item()
                ))
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
        if self.predict_head:
            query_targets = ["tail", "head"]
        else:
            query_targets = ["tail"]
        with torch.no_grad(), tqdm(total=len(tkg.test_set) * len(query_targets)) as pbar:
            results = []
            for query_target in query_targets:
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
                        query_target=query_target,
                    )
                    # Generate results
                    predictions = self.predict(
                        prompt=prompt,
                        candidates=candidates,
                    )
                    result = QueryResult(
                        query=query,
                        query_target=query_target,
                        predictions=predictions,
                        candidates=candidates,
                        prompt=prompt,
                        answer=query.tail,
                    )
                    result.time_filter(time_filter_set)
                    results.append(result)
                    metrics = metric(results)
                    pbar.set_description(f"Hit@10: {metrics['hit@10']:.2%}")
                    pbar.update()
        metrics = metric(results)
        print(metrics)
