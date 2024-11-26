import itertools
from typing import Dict, Any, Tuple, List

import numpy as np


class RuleLearner:
    """
    Rule learner.
    """
    def __init__(
            self,
            rel_indexing: Dict[int, np.ndarray],
            num_samples: int = 500
        ):
        self.rel_indexing = rel_indexing
        self.num_samples = num_samples
        self.found_rules = []
        self.rules_dict = {}

    def create_rule(self, walk: Dict[str, Any]):
        """
        Create a rule given a temporal random walk.
        """
        rule = {
            "head_rel": int(walk["relations"][0]),
            "body_rel": int(walk["relations"][1]),
        }
        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            conf, rule_support, body_support = self.estimate_confidence(rule)
            rule["conf"] = conf
            rule["rule_supp"] = rule_support
            rule["body_supp"] = body_support

            if rule["conf"]:
                self.rules_dict.setdefault(rule["head_rel"], []).append(rule)

    def estimate_confidence(
            self,
            rule: Dict[str, Any],
    ) -> Tuple[float, int, int]:
        """
        Estimate the confidence of a rule by sampling bodies and check support.
        """
        # Sample bodies
        all_bodies = []
        for _ in range(self.num_samples):
            body_ents_tss = self.sample_body(rule["body_rel"])
            all_bodies.append(body_ents_tss)
        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        # Check rule support
        if body_support:
            rule_support = self.calculate_rule_support(
                unique_bodies=unique_bodies,
                head_rel=rule["head_rel"],
            )
            confidence = round(rule_support / body_support, 6)
        else:
            confidence, rule_support = 0., 0
        return confidence, rule_support, body_support

    def sample_body(self, body_rel: int) -> List:
        """
        Try to sample a walk according to rule body.
        """
        sample_successful = True
        body_ents_tss = []
        facts = self.rel_indexing[body_rel]
        sampled_fact = facts[np.random.choice(len(facts))]
        body_subj = sampled_fact[0]
        body_obj = sampled_fact[2]
        body_ts = sampled_fact[3]
        body_ents_tss.append(body_subj)
        body_ents_tss.append(body_ts)
        body_ents_tss.append(body_obj)
        return body_ents_tss

    def calculate_rule_support(
            self,
            unique_bodies: List[List[int]],
            head_rel: int,
    ) -> int:
        """
        Calculate rule support.
        """
        rule_support = 0
        head_rel_facts = self.rel_indexing[head_rel]
        for body in unique_bodies:
            check = (
                (head_rel_facts[:, 0] == body[0]) &
                (head_rel_facts[:, 2] == body[2]) &
                (head_rel_facts[:, 3] > body[1])
            )
            if True in check:
                rule_support += 1
        return rule_support
