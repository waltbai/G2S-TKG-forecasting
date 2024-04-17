import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm4tkg.evaluation import metric, QueryResult, Prediction
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
            anonymize: bool = False,
            anonymize_time: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            truncation_side="left",
            # padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )
        self.backbone = backbone
        self.model.to(device)
        self.device = device
        self.history_length = history_length
        self.history_type = history_type
        self.history_direction = history_direction
        self.top_k = top_k
        self.predict_head = predict_head
        self.anonymize = anonymize
        self.anonymize_time = anonymize_time

    def evaluate(
            self,
            tkg: TemporalKG,
    ) -> None:
        """Evaluate on test set."""
        self.model.eval()
        # Prepare time filter set
        time_filter_set = set()
        for fact in tkg.test_set:
            time_filter_set.add(
                (fact.head, fact.rel, fact.tail, fact.time)
            )
        if self.predict_head:
            query_targets = ["tail", "head"]
        else:
            query_targets = ["tail"]
        # Prepare input
        results = []
        tot_num = len(tkg.test_set) * len(query_targets)
        with tqdm(total=tot_num) as pbar:
            pbar.set_description("Search history and construct prompt")
            for query_target in query_targets:
                for query in tkg.test_set:
                    prompt, candidates = quadruple_prompt(
                        query=query,
                        tkg=tkg,
                        history_length=self.history_length,
                        history_type=self.history_type,
                        history_direction=self.history_direction,
                        anonymize=self.anonymize,
                        anonymize_time=self.anonymize_time,
                        shuffle=False,
                        query_target=query_target,
                    )
                    results.append(QueryResult(
                        query=query,
                        query_target=query_target,
                        predictions=[],
                        candidates=candidates,
                        prompt=prompt,
                        answer=query.tail if query_target == "tail" else query.head,
                    ))
                    pbar.update()
        # Collect predictions
        probs = []
        probs_idx = []
        with torch.no_grad(), tqdm(total=tot_num) as pbar:
            pbar.set_description("Predict queries")
            for i in range(tot_num):
                inputs = self.tokenizer(
                    results[i].prompt,
                    return_tensors="pt",
                    truncation=True,
                    # padding=True,
                ).to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    renormalize_logits=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                single_probs, single_probs_idx = torch.sort(
                    outputs.scores[0], dim=-1, descending=True
                )
                single_probs = single_probs[0, :self.top_k].to("cpu")
                single_probs_idx = single_probs_idx[0, :self.top_k].to("cpu")
                probs.append(single_probs)
                probs_idx.append(single_probs_idx)
                pbar.update()
        # Decode predictions
        with tqdm(total=tot_num) as pbar:
            pbar.set_description("Decode results")
            for i in range(tot_num):
                query_predictions = []
                predict_ids = set()
                for score, token_id in zip(probs[i], probs_idx[i]):
                    ent_id = self.tokenizer.decode(token_id).strip()
                    if ent_id in results[i].candidates and ent_id not in predict_ids:
                        predict_ids.add(ent_id)
                        entity = results[i].candidates[ent_id]
                        if self.anonymize:
                            entity = tkg.entities[int(entity)]
                        query_predictions.append(Prediction(
                            entity=entity,
                            score=score.item(),
                        ))
                results[i].predictions = query_predictions
                results[i].time_filter(time_filter_set)
                pbar.update()
        # Compute metrics
        metrics = metric(results)
        print(f"Experiment settings:\n"
              f"Model: {self.backbone}-ICL\n"
              f"Dataset: {tkg.dataset_name}\n"
              f"Prompt: quadruple\n"
              f"Anonymize: {self.anonymize}\n"
              f"Anonymize time: {self.anonymize_time}\n"
              f"Predictions: {self.top_k}\n"
              f"History length: {self.history_length}\n"
              f"History type: {self.history_type}\n"
              f"History direction: {self.history_direction}\n\n")
        print(f"Hit@1:  {metrics['hit@1']:.2%}\n"
              f"Hit@3:  {metrics['hit@3']:.2%}\n"
              f"Hit@10: {metrics['hit@10']:.2%}\n")
