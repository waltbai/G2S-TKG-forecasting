import logging

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict

from src.evaluation import metric, Query
from src.preprocess.tkg import TemporalKG
from src.prompt import quadruple_prompt
from src.utils.common import format_params


class InContextLearningModel:
    """In-context learning model without supervised fine-tuning.

    This is a re-implementation of (Lee, et al., emnlp-2023).
    """

    def __init__(
            self,
            backbone: str = "openai-community/gpt2",
            device: str = "cpu",
            # fp16: bool = False,
            history_length: int = 30,
            history_type: str = "entity",
            history_direction: str = "uni",
            top_k: int = 30,
            predict_head: bool = True,
            time_filter: bool = False,
            filter_duplicate: bool = False,
            pbar: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            truncation_side="left",
            # padding_side="left",
        )
        # self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone,
            # torch_dtype=torch.float16 if fp16 else torch.float32,
        )
        self.backbone = backbone
        self.model_name = self.backbone.split("/")[-1] + "-ICL"
        self.model.to(device)
        self.device = device
        self.history_length = history_length
        self.history_type = history_type
        self.history_direction = history_direction
        self.top_k = top_k
        self.predict_head = predict_head
        self.time_filter = time_filter
        self.pbar = pbar
        self.filter_duplicate = filter_duplicate
        self.logger = logging.getLogger("ICLModel")

    def experiment_settings(
            self,
            tkg: TemporalKG,
    ) -> str:
        """Experiment settings."""
        params = [
            ("Model", self.model_name),
            ("Dataset", tkg.dataset_name),
            ("Anonymize entity", tkg.anon_entity),
            ("Anonymize rel", tkg.anon_rel),
            ("Anonymize time", tkg.anon_time),
            ("# Predictions", self.top_k),
            ("History length", self.history_length),
            ("History type", self.history_type),
            ("History direction", self.history_length),
            ("Time filter", self.time_filter),
        ]
        return format_params(params)

    @staticmethod
    def experiment_results(
            metrics: Dict[str, float],
    ) -> str:
        """Experiment results."""
        params = [
            ("Hit@1", f"{metrics['hit@1']:.2%}"),
            ("Hit@3", f"{metrics['hit@3']:.2%}"),
            ("Hit@10", f"{metrics['hit@10']:.2%}"),
        ]
        return format_params(params)

    def predict(
            self,
            query: Query,
            tkg: TemporalKG,
    ):
        """Predict a query, fill in the predictions in-place."""
        # Inference
        inputs = self.tokenizer(
            query.prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        probs, probs_idx = torch.sort(
            outputs.scores[0], dim=-1, descending=True
        )
        probs = probs[0, :self.top_k]
        probs_idx = probs_idx[0, :self.top_k]
        # Decode predictions
        predictions = []
        scores = []
        predict_ids = set()
        for score, token_id in zip(probs, probs_idx):
            ent_id = self.tokenizer.decode(token_id).strip()
            if ent_id in query.candidates:
                if self.filter_duplicate and ent_id in predict_ids:
                    continue
                predict_ids.add(ent_id)
                entity = tkg.deanonymize_entity(query.candidates[ent_id])
                predictions.append(entity)
                scores.append(score.item())
        query.predictions = predictions
        query.scores = scores

    def evaluate(
            self,
            tkg: TemporalKG,
    ) -> None:
        """Evaluate on test set."""
        self.model.eval()
        # Prepare input
        tot_num = len(tkg.test_queries)
        tqdm_params = {
            "total": tot_num,
            "ascii": True,
            "disable": not self.pbar,
        }
        self.logger.info("Search history and construct prompts.")
        with tqdm(**tqdm_params) as pbar:
            for query in tkg.test_queries:
                prompt, candidates = quadruple_prompt(
                    query=query,
                    tkg=tkg,
                    history_length=self.history_length,
                    history_type=self.history_type,
                    history_direction=self.history_direction,
                    shuffle=False,
                )
                query.prompt = prompt
                query.candidates = candidates
                pbar.update()
        # Collect predictions
        self.logger.info("Predict queries.")
        with torch.no_grad(), tqdm(**tqdm_params) as pbar:
            for i in range(tot_num):
                self.predict(tkg.test_queries[i], tkg)
                pbar.update()
        # Compute metrics
        metrics = metric(
            queries=tkg.test_queries,
            time_filter=self.time_filter,
        )
        self.logger.info(
            f"Experimental settings:\n{self.experiment_settings(tkg)}"
        )
        self.logger.info(
            f"Experimental Results:\n{self.experiment_results(metrics)}"
        )
