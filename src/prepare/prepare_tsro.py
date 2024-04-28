"""TSRO prompt means each fact is in the format "t:[s,r,o]"."""

import argparse
import json
import logging
import os

from typing import List

from tqdm import tqdm

from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import TKG


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data", type=str)
    parser.add_argument("--dataset", default="ICEWS14", type=str,
                        choices=["ICEWS14", "ICEWS05-15", "ICEWS18", "WIKI", "YAGO"])
    parser.add_argument("--output_dir", default="data", type=str)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_evaluate", default=False, action="store_true")
    # TODO: update to BooleanOptionalAction when python>=3.9
    parser.add_argument("--do_predict", default=True, action="store_true")
    parser.add_argument("--no-do_predict",
                        dest="do_predict", action="store_false")
    parser.add_argument("--anonymize_entity", default=False, action="store_true")
    parser.add_argument("--anonymize_rel", default=False, action="store_true")
    parser.add_argument("--anonymize_time", default=True)
    parser.add_argument("--no-anonymize_time",
                        dest="anonymize_time", action="store_false")
    parser.add_argument("--history_length", default=30, type=int)
    parser.add_argument("--history_type", default="entity", type=str)
    parser.add_argument("--history_direction", default="uni", type=str)
    return parser.parse_args()


def construct_queries(facts: List[Fact]) -> List[Query]:
    """Construct queries from dataset."""
    queries = []
    answers = {}
    for entity_role in ["head", "tail"]:
        for fact in facts:
            query_rel = fact.rel
            query_time = fact.time
            if entity_role == "head":
                query_entity = fact.head
                answer = fact.tail
            else:  # entity_role == "tail"
                query_entity = fact.tail
                answer = fact.head
            key = (query_entity, query_rel, query_time, entity_role)
            answers.setdefault(key, []).append(answer)
            queries.append(
                Query(
                    entity=query_entity,
                    rel=query_rel,
                    answer=answer,
                    time=query_time,
                    entity_role=entity_role,
                )
            )
    # Sort queries by time
    for query in queries:
        key = (query.entity, query.rel, query.time, query.entity_role)
        query.filters = [_ for _ in answers[key] if _ != query.answer]
    queries = sorted(queries, key=lambda x: x.time)
    return queries


def construct_prompt(
        query: Query,
        tkg: TKG,
        history_length: int = 30,
        history_type: str = "entity",
        history_direction: str = "uni",
        anonymize_entity: bool = False,
        anonymize_rel: bool = False,
        anonymize_time: bool = True,
):
    """Construct prompts for queries."""
    # Determine key of searching history
    if history_direction == "uni":
        if history_type == "entity":
            search_key_1 = query.entity_role
            search_key_2 = query.entity
        else:   # history_type == "pair"
            search_key_1 = f"{query.entity_role}_rel"
            search_key_2 = (query.entity, query.rel)
    else:   # history_direction == "bi"
        if history_type == "entity":
            search_key_1 = "both"
            search_key_2 = query.entity
        else:   # history_type == "pair"
            search_key_1 = "both_rel"
            search_key_2 = (query.entity, query.rel)
    facts = tkg.search_history[search_key_1][search_key_2]
    history = [fact for fact in facts if fact.time < query.time]
    history = history[-history_length:]
    # Convert to quadruple and anonymize
    his_quads = []
    for fact in history:
        quad = fact.quadruple() if fact.head == query.entity \
            else fact.quadruple("swap")
        head, rel, tail, time = quad
        if anonymize_entity:
            head = str(tkg.entity2id[head])
            tail = str(tkg.entity2id[tail])
        if anonymize_rel:
            rel = str(tkg.relation2id[rel])
        if anonymize_time:
            time = str(tkg.time2id[time])
        his_quads.append([head, rel, tail, time])
    # Count candidate frequency and sort
    candidate_freq = {}
    for quad in his_quads:
        candidate_freq.setdefault(quad[2], 0)
        candidate_freq[quad[2]] += 1
    candidate_sorted = list(
        sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
    )
    # Re-map candidates to IDs, start from 0
    candidate_mapping = {}
    for i, (entity, _) in enumerate(candidate_sorted):
        candidate_mapping[entity] = i
    # Construct prompt
    prompt = ""
    for quad in his_quads:
        head, rel, tail, time = quad
        # TODO: Original paper uses comma (","), however,
        #  to distinguish from the comma in relation name,
        #  we will change to semicolon (";")
        prompt += f"{time}:[{head},{rel},{candidate_mapping[tail]}.{tail}]\n"
    entity = query.entity
    rel = query.rel
    time= query.time
    answer = query.answer
    filters = query.filters
    if anonymize_entity:
        entity = str(tkg.entity2id[entity])
        answer = str(tkg.entity2id[answer])
        filters = [str(tkg.entity2id[_]) for _ in filters]
    if anonymize_rel:
        rel = str(tkg.relation2id[rel])
    if anonymize_time:
        time = str(tkg.time2id[time])
    prompt += f"{time}:[{entity},{rel},"
    candidates = {str(v): k for k, v in candidate_mapping.items()}
    return {
        "prompt": prompt,
        "answer": answer,
        "filters": filters,
        "candidates": candidates,
    }


if __name__ == "__main__":
    args = get_args()
    logger = logging.getLogger("prepare.tsro")
    logger.info("Load TKG.")
    tkg = TKG.load(
        dataset_dir=args.dataset_dir,
        dataset=args.dataset,
    )
    logger.info("Construct queries.")
    queries = {
        "train": construct_queries(tkg.train_facts) if args.do_train else [],
        "valid": construct_queries(tkg.valid_facts) if args.do_evaluate else [],
        "test": construct_queries(tkg.test_facts) if args.do_predict else []
    }
    logger.info("Prepare inputs.")
    default_params = {
        "history_length": args.history_length,
        "history_type": args.history_type,
        "history_direction": args.history_direction,
        "anonymize_entity": args.anonymize_entity,
        "anonymize_rel": args.anonymize_rel,
        "anonymize_time": args.anonymize_time,
    }
    use_tqdm = True
    tqdm_params = {
        "ascii": False, "disable": not use_tqdm
    }
    dataset = {}
    for key in queries:
        total_num = len(queries[key])
        if total_num:
            dataset[key] = {
                "prompt": [],
                "answer": [],
                "filters": [],
                "candidates": [],
            }
            with tqdm(total=total_num, **tqdm_params) as pbar:
                pbar.set_description(f"Process {key} queries")
                for query in queries[key]:
                    result = construct_prompt(query, tkg, **default_params)
                    dataset[key]["prompt"].append(result["prompt"])
                    dataset[key]["answer"].append(result["answer"])
                    dataset[key]["filters"].append(result["filters"])
                    dataset[key]["candidates"].append(result["candidates"])
                    pbar.update()
    dataset_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_TSRO.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    logger.info(f"Dataset save to {dataset_path}.")
