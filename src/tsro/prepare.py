import json
import logging
import os
from typing import List, Dict, Any

__all__ = [
    "PrepareArguments",
    "get_data_name",
    "prepare"
]

from tqdm import tqdm

from src.tsro.args import PrepareArguments
from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import TKG

logger = logging.getLogger(__name__)


def get_data_name(
        prepare_args: PrepareArguments,
        prepare_parts: List[str] = None
) -> str:
    """Generate data file name via arguments."""
    name = f"{prepare_args.dataset}-tsro"
    prepare_parts = prepare_parts or ["train", "valid", "test"]
    name += "-" + "-".join(prepare_parts)
    if prepare_args.anonymize_entity:
        name += "-anonymize_entity"
    if prepare_args.anonymize_rel:
        name += "-anonymize_rel"
    if prepare_args.anonymize_time:
        name += "-anonymize_time"
    name += f"-{prepare_args.history_length}"
    name += f"-{prepare_args.history_type}"
    name += f"-{prepare_args.history_direction}"
    name += ".json"
    return name


def prepare(
        args: PrepareArguments,
        prepare_parts: List[str] = None,
        prepare_path: str = None,
        use_tqdm: bool = True,
):
    """Dataset preparation function."""
    prepare_parts = prepare_parts or ["train", "valid", "test"]
    prepare_path = prepare_path or os.path.join(
        args.prepare_dir, get_data_name(args, prepare_parts)
    )
    os.makedirs(args.prepare_dir, exist_ok=True)

    # Load TKG
    logger.info("Load TKG.")
    tkg = TKG.load(
        dataset_dir=args.dataset_dir,
        dataset=args.dataset,
    )

    # Construct queries
    logger.info("Construct Queries.")
    queries = {
        "train": construct_queries(tkg.train_facts) if "train" in prepare_parts else [],
        "valid": construct_queries(tkg.valid_facts) if "valid" in prepare_parts else [],
        "test": construct_queries(tkg.test_facts) if "test" in prepare_parts else []
    }

    # Construct prompts
    params = {
        "history_length": args.history_length,
        "history_type": args.history_type,
        "history_direction": args.history_direction,
        "anonymize_entity": args.anonymize_entity,
        "anonymize_rel": args.anonymize_rel,
        "anonymize_time": args.anonymize_time,
    }
    tqdm_params = {"disable": not use_tqdm}
    ds = {}
    for key in queries:
        total_num = len(queries[key])
        ds[key] = {
            "prompt": [],
            "answer": [],
            "filters": [],
            "candidates": [],
        }
        with tqdm(total=total_num, **tqdm_params) as pbar:
            pbar.set_description(f"Process {key} queries")
            for query in queries[key]:
                result = construct_prompt(
                    query=query,
                    tkg=tkg,
                    history_length=args.history_length,
                    history_type=args.history_type,
                    history_direction=args.history_direction,
                    anonymize_entity=args.anonymize_entity,
                    anonymize_rel=args.anonymize_rel,
                    anonymize_time=args.anonymize_time,
                )
                ds[key]["prompt"].append(result["prompt"])
                ds[key]["answer"].append(result["answer"])
                ds[key]["filters"].append(result["filters"])
                ds[key]["candidates"].append(result["candidates"])
                pbar.update()
    with open(prepare_path, "w") as f:
        json.dump(ds, f)
    logger.info(f"Dataset save to {prepare_path}.")


def construct_queries(facts: List[Fact]) -> List[Query]:
    """Construct queries from TKG."""
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
) -> Dict[str, Any]:
    """Construct prompts for queries."""
    # Determine key of searching history
    if history_direction == "uni":
        if history_type == "entity":
            search_key_1 = query.entity_role
            search_key_2 = query.entity
        else:  # history_type == "pair"
            search_key_1 = f"{query.entity_role}_rel"
            search_key_2 = (query.entity, query.rel)
    else:  # history_direction == "bi"
        if history_type == "entity":
            search_key_1 = "both"
            search_key_2 = query.entity
        else:  # history_type == "pair"
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
    time = query.time
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
    # Avoid that answer is not in the candidate list
    # candidate_mapping.setdefault(answer, len(candidate_mapping))
    prompt += f"{time}:[{entity},{rel},"
    candidates = {str(v): k for k, v in candidate_mapping.items()}
    if answer not in candidate_mapping:
        answer = str(len(candidates))
        candidates.setdefault(answer, None)
    else:
        answer = str(candidate_mapping[answer])
    return {
        "prompt": prompt,
        "answer": answer,
        "filters": filters,
        "candidates": candidates,
    }
