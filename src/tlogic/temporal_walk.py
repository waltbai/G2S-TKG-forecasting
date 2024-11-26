import numpy as np
from typing import Dict, Tuple


def cache_by_rel(facts: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Cache all facts by indexing relations.
    """
    cached_facts = {}
    relations = list(set(facts[:, 1]))
    for rel in relations:
        cached_facts[rel] = facts[facts[:, 1] == rel]
    return cached_facts


def cache_by_subject(facts: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Cache all facts by indexing subject entity.
    """
    cached_facts = {}
    subjs = list(set(facts[:, 0]))
    for subj in subjs:
        cached_facts[subj] = facts[facts[:, 0] == subj]
    return cached_facts

class TemporalWalkSampler:
    """
    Temporal Walk Sampler.
    """

    def __init__(self, facts):
        self.facts = facts
        self.rel_indexing = cache_by_rel(facts)
        self.subj_indexing = cache_by_subject(facts)

    def sample_head_fact(self, rel_idx) -> np.ndarray:
        """
        Uniformly sample head fact.
        """
        facts = self.rel_indexing[rel_idx]
        head_fact = facts[np.random.choice(len(facts))]
        return head_fact

    def sample_body_fact(self, head_fact: np.ndarray) -> np.ndarray:
        """
        Sample body fact.
        """
        subj_idx = int(head_fact[0])
        obj_idx = int(head_fact[2])
        cur_ts = int(head_fact[3])
        # Filter by subject entity
        filtered_facts = self.subj_indexing[subj_idx]
        # Filter by object entity and time
        filtered_facts = filtered_facts[
            (filtered_facts[:, 2] == obj_idx) &
            (filtered_facts[:, 3] < cur_ts)
        ]
        if len(filtered_facts):
            # Sample by exp distribution
            tss = filtered_facts[:, 3]
            prob = np.exp(tss - cur_ts)
            try:
                prob = prob/ np.sum(prob)
                body_fact = filtered_facts[
                    np.random.choice(range(len(filtered_facts)), p=prob)
                ]
            except ValueError: # All timestamps are far away
                body_fact = filtered_facts[np.random.choice(len(filtered_facts))]
        else:
            body_fact = []
        return body_fact

    def sample_walk(self, rel_idx: int) -> Tuple[bool, Dict]:
        """
        Try to sample a cyclic temporal random walk.
        """
        walk_successful = True
        walk = {}
        # Head
        head_fact = self.sample_head_fact(rel_idx)
        subj_idx = int(head_fact[0])
        obj_idx = int(head_fact[2])
        cur_ts = int(head_fact[3])
        walk["entities"] = [subj_idx, obj_idx]
        walk["relations"] = [int(head_fact[1])]
        walk["timestamps"] = [cur_ts]

        # Body
        body_fact = self.sample_body_fact(head_fact)
        if len(body_fact):
            walk["entities"].append(subj_idx)
            walk["relations"].append(int(body_fact[1]))
            walk["timestamps"].append(int(body_fact[3]))
        else:
            walk_successful = False

        return walk_successful, walk
