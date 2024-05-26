from abc import ABC, abstractmethod

from typing import List

from tqdm import tqdm

from src.utils.query import Query


class PromptConstructStrategy(ABC):
    """Prompt construct strategy base class."""

    def __init__(self, use_tqdm: bool = False):
        self.use_tqdm = use_tqdm

    def __call__(
            self,
            queries: List[Query],
            sep: str = ","
    ):
        with tqdm(total=len(queries), disable=not self.use_tqdm) as pbar:
            for query in queries:
                self.construct(query, sep)
                pbar.update()

    @abstractmethod
    def construct(self, query: Query, sep: str = ","):
        """Construct prompt for each query."""


class InlinePromptConstructStrategy(PromptConstructStrategy):
    """Query is given inline, no special strings for history and query.

    Example:
        ...
        t_i:[s_i,r_i,o_i]
        ...
        t_q:[s_q,r_q,
    """

    def construct(self, query: Query, sep: str = ","):
        query_entity = query.entity_mapping[query.entity]
        query_rel = query.rel_mapping[query.rel]
        query_answer = query.entity_mapping[query.answer] \
            if query.answer in query.entity_mapping else None
        query_time = query.time_mapping[query.time]

        # Convert historical facts into quadruples
        map_quad = lambda x: [
            query.entity_mapping[x.head],
            query.rel_mapping[x.rel],
            query.entity_mapping[x.tail],
            query.time_mapping[x.time]
        ]
        his_quads = []
        for fact in query.history:
            quad = map_quad(fact)
            # Swap head and tail if tail is query_entity
            if quad[2] == query_entity:
                quad = [quad[2], quad[1], quad[0], quad[3]]
            his_quads.append(quad)

        # Count candidate frequency and sort
        candidate_freq = {}
        for quad in his_quads:
            candidate_freq.setdefault(quad[2], 0)
            candidate_freq[quad[2]] += 1
        candidate_sorted = list(
            sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
        )

        # Re-map candidates to IDs, start from 0
        cand_mapping = {}
        for i, (entity, _) in enumerate(candidate_sorted):
            cand_mapping[entity] = str(i)

        # Construct prompt
        prompt = ""
        for quad in his_quads:
            head, rel, tail, time = quad
            prompt += f"{time}:[{head}{sep}{rel}{sep}{cand_mapping[tail]}.{tail}]\n"
        prompt += f"{query_time}:[{query_entity}{sep}{query_rel}{sep}"
        candidates = {str(v): k for k, v in cand_mapping.items()}
        if query_answer not in cand_mapping:
            label = str(len(candidates))
            candidates.setdefault(label, None)
        else:
            label = cand_mapping[query_answer]

        query.prompt = prompt
        query.candidates = candidates
        query.label = label
        query.anonymous_filters = [
            query.entity_mapping[ent] for ent in query.filters
            if ent in query.entity_mapping
        ]


class QAPromptConstructStrategy(PromptConstructStrategy):
    """Query is given with QA form.

    Example:
        ### History ###
        ...
        t_i:[s_i,r_i,o_i]
        ...

        ### Query ###
        t_q:[s_q,r_q,?]

        ### Answer ###
    """

    def construct(self, query: Query, sep: str = ","):
        query_entity = query.entity_mapping[query.entity]
        query_rel = query.rel_mapping[query.rel]
        query_answer = query.entity_mapping[query.answer] \
            if query.answer in query.entity_mapping else None
        query_time = query.time_mapping[query.time]

        # Convert historical facts into quadruples
        map_quad = lambda x: [
            query.entity_mapping[x.head],
            query.rel_mapping[x.rel],
            query.entity_mapping[x.tail],
            query.time_mapping[x.time]
        ]
        his_quads = []
        for fact in query.history:
            quad = map_quad(fact)
            # Swap head and tail if tail is query_entity
            if quad[2] == query_entity:
                quad = [quad[2], quad[1], quad[0], quad[3]]
            his_quads.append(quad)

        # Count candidate frequency and sort
        candidate_freq = {}
        for quad in his_quads:
            candidate_freq.setdefault(quad[2], 0)
            candidate_freq[quad[2]] += 1
        candidate_sorted = list(
            sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
        )

        # Re-map candidates to IDs, start from 0
        cand_mapping = {}
        for i, (entity, _) in enumerate(candidate_sorted):
            cand_mapping[entity] = str(i)

        # Construct prompt
        prompt = "### History ###\n"
        for quad in his_quads:
            head, rel, tail, time = quad
            prompt += f"{time}:[{head}{sep}{rel}{sep}{cand_mapping[tail]}.{tail}]\n"
        prompt += f"\n### Query ###\n"
        prompt += f"{query_time}:[{query_entity}{sep}{query_rel}{sep}?]\n"
        prompt += f"\n### Answer ###\n"
        candidates = {str(v): k for k, v in cand_mapping.items()}
        if query_answer not in cand_mapping:
            label = str(len(candidates))
            candidates.setdefault(label, None)
        else:
            label = cand_mapping[query_answer]

        query.prompt = prompt
        query.candidates = candidates
        query.label = label
        query.anonymous_filters = [
            query.entity_mapping[ent] for ent in query.filters
            if ent in query.entity_mapping
        ]


def get_prompt_constructor(
        strategy: str,
        use_tqdm: bool = False,
) -> PromptConstructStrategy:
    """Get prompt constructor by name."""
    if strategy == "inline":
        return InlinePromptConstructStrategy(use_tqdm)
    else:   # strategy == "qa"
        return QAPromptConstructStrategy(use_tqdm)
