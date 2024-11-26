"""Main algorithm are copy from TLogic.

The algorithm is simplified to one-hop rules.
"""
import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
from joblib import Parallel, delayed

from src.tlogic.rule_learning import RuleLearner
from src.tlogic.temporal_walk import TemporalWalkSampler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data", type=str)
    parser.add_argument("--dataset", default="ICEWS14", type=str)
    parser.add_argument("--num_walks", default=100, type=int)
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="data/rules", type=str)
    return parser.parse_args()


def learn_rules(
        proc_id: int,
        num_relations: int,
        all_relations: List[int],
        num_walks: int,
        temporal_walk_sampler: TemporalWalkSampler,
        rule_learner: RuleLearner,
        seed: int = 42,
) -> Dict[str, Any]:
    """
    Learn rules.
    """
    if seed:
        np.random.seed(seed)

    relations_idx = range(
        proc_id * num_relations,
        min(len(all_relations), (proc_id+1) * num_relations)
    )

    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for _ in range(num_walks):
            walk_successful, walk = temporal_walk_sampler.sample_walk(rel)
            if walk_successful:
                rule_learner.create_rule(walk)
        num_rules.append(sum([len(v) for k, v in rule_learner.rules_dict.items()]) // 2)
        num_new_rules = num_rules[-1] - num_rules[-2]
        print(
            "Process {0}: relation {1}/{2}, {3} rules".format(
                proc_id,
                k - relations_idx[0] + 1,
                len(relations_idx),
                num_new_rules,
            )
        )

    return rule_learner.rules_dict


def main(
        dataset_dir: str = "data",
        dataset: str = "ICEWS14",
        num_walks: int = 100,
        num_processes: int = 1,
        seed: int = 42,
        output_dir: str = "data/rules",
):
    """
    Main learning function.
    """
    # Load train fact indices
    train_path = os.path.join(dataset_dir, dataset, "train.txt")
    train_facts = []
    with open(train_path, "r") as f:
        for line in f:
            s, p, o, t = map(int, line.strip().split("\t")[:4])
            train_facts.append([s, p, o, t])
    train_facts = np.array(train_facts)

    # Create temporal walk sampler
    temporal_walk_sampler = TemporalWalkSampler(
        facts=train_facts
    )

    # Create rule learner
    rule_learner = RuleLearner(rel_indexing=temporal_walk_sampler.rel_indexing)

    # Multiprocessing rule learning
    all_relations = sorted(temporal_walk_sampler.rel_indexing)
    num_relations = math.ceil(len(all_relations) / num_processes)
    output = Parallel(n_jobs=num_processes)(
        delayed(learn_rules)(
            proc_id=proc_id,
            num_relations=num_relations,
            all_relations=all_relations,
            num_walks=num_walks,
            temporal_walk_sampler=temporal_walk_sampler,
            rule_learner=rule_learner,
            seed=seed,
        )
        for proc_id in range(num_processes)
    )

    # Process rules
    all_rules = output[0]
    for i in range(1, num_processes):
        all_rules.update(output[i])
    rules = all_rules

    # Map relation ID to names
    relation2id_path = os.path.join(dataset_dir, dataset, "relation2id.txt")
    relation2id = {}
    with open(relation2id_path, "r") as f:
        for line in f:
            name, idx = line.strip().split("\t")
            relation2id[name] = int(idx)
    id2relation = {v: k for k, v in relation2id.items()}
    processed_rules = {}
    for key in rules:
        for rule_item in rules[key]:
            head_idx = rule_item["head_rel"]
            body_idx = rule_item["body_rel"]
            head = id2relation[head_idx]
            body = id2relation[body_idx]
            confidence = rule_item["conf"]
            processed_rules.setdefault(head, {}).setdefault(body, confidence)
    
    sorted_rules = {}
    for head in processed_rules:
        for body, confidence in sorted(processed_rules[head].items(), key=lambda x:x[1], reverse=True):
            sorted_rules.setdefault(head, {}).setdefault(body, confidence)
    
    output_path = os.path.join(output_dir, f"{dataset}.json")
    with open(output_path, "w") as f:
        json.dump(sorted_rules, f)


if __name__ == "__main__":
    args = get_args()
    main(
        dataset_dir=args.dataset_dir,
        dataset=args.dataset,
        num_walks=args.num_walks,
        num_processes=args.num_processes,
        seed=args.seed,
    )
