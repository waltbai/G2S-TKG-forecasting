from abc import ABC, abstractmethod

from typing import List, Dict

from tqdm import tqdm

from src.utils.query import Query


def relabel_candidates(his_quads: List[List[str]]) -> Dict[str, str]:
    """Relabel candidates by frequency."""
    # Count candidate frequency and sort
    candidate_freq = {}
    for quad in his_quads:
        candidate_freq.setdefault(quad[2], 0)
        candidate_freq[quad[2]] += 1
    candidate_sorted = list(
        sorted(candidate_freq.items(), key=lambda x: x[1], reverse=True)
    )

    # Relabel candidates to IDs, start from 0
    cand_mapping = {}
    for i, (entity, _) in enumerate(candidate_sorted):
        cand_mapping[entity] = str(i)

    return cand_mapping


def original_id(entity_mapping: Dict[str, str]) -> Dict[str, str]:
    """Use original id."""
    cand_mapping = {}
    # Use original id and remove prefix
    for value in entity_mapping.values():
        cand_mapping[value] = value.replace("ENT_", "")

    return cand_mapping


class PromptConstructStrategy(ABC):
    """Prompt construct strategy base class."""

    def __init__(
            self,
            prefix: bool = False,
            cand_relabel: bool = True,
            use_tqdm: bool = False
    ):
        self.prefix = prefix
        self.cand_relabel = cand_relabel
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
    def construct_prompt(
            self,
            query_quad: List[str],
            his_quads: List[List[str]],
            cand_mapping: Dict[str, str],
            query: Query,
            sep: str = ",",
    ):
        """Main func of prompt construction."""

    def construct(self, query: Query, sep: str = ","):
        """Construct prompt for each query."""
        # Query
        query_entity = query.entity_mapping[query.entity]
        query_rel = query.rel_mapping[query.rel]
        query_answer = query.entity_mapping[query.answer] \
            if query.answer in query.entity_mapping else None
        query_time = query.time_mapping[query.time]
        query_quad = [query_entity, query_rel, query_answer, query_time]

        # Historical facts
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

        # Candidate mapping
        if self.cand_relabel:
            cand_mapping = relabel_candidates(his_quads)
        else:
            cand_mapping = original_id(query.entity_mapping)

        # Construct prompt
        self.construct_prompt(
            query_quad=query_quad,
            his_quads=his_quads,
            cand_mapping=cand_mapping,
            query=query,
            sep=sep,
        )


class InlinePromptConstructStrategy(PromptConstructStrategy):
    """Query is given inline, no special strings for history and query.

    Example:
        ...
        t_i:[s_i,r_i,o_i]
        ...
        t_q:[s_q,r_q,
    """

    def construct_prompt(
            self,
            query_quad: List[str],
            his_quads: List[List[str]],
            cand_mapping: Dict[str, str],
            query: Query,
            sep: str = ",",
    ):
        """Main func of prompt construction."""
        query_entity, query_rel, query_answer, query_time = query_quad

        # Construct prompt
        prompt = ""
        for quad in his_quads:
            head, rel, tail, time = quad
            if self.cand_relabel:
                prompt += f"{time}:[{head}{sep}{rel}{sep}{cand_mapping[tail]}.{tail}]\n"
            else:
                prompt += f"{time}:[{head}{sep}{rel}{sep}{tail}]\n"
        prompt += f"{query_time}:[{query_entity}{sep}{query_rel}{sep}"
        if self.prefix:
            prompt += "ENT_"
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

    def construct_prompt(
            self,
            query_quad: List[str],
            his_quads: List[List[str]],
            cand_mapping: Dict[str, str],
            query: Query,
            sep: str = ",",
    ):
        """Main func of prompt construction."""
        query_entity, query_rel, query_answer, query_time = query_quad

        # Construct prompt
        prompt = "### History ###\n"
        for quad in his_quads:
            head, rel, tail, time = quad
            if self.cand_relabel:
                prompt += f"{time}:[{head}{sep}{rel}{sep}{cand_mapping[tail]}.{tail}]\n"
            else:
                prompt += f"{time}:[{head}{sep}{rel}{sep}{tail}]\n"
        prompt += f"\n### Query ###\n"
        prompt += f"{query_time}:[{query_entity}{sep}{query_rel}{sep}?]\n"
        prompt += f"\n### Answer ###\n"
        if self.prefix:
            prompt += "ENT_"
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
        prefix: bool = False,
        cand_relabel: bool = True,
        use_tqdm: bool = False,
) -> PromptConstructStrategy:
    """Get prompt constructor by name."""
    if strategy == "inline":
        return InlinePromptConstructStrategy(
            prefix=prefix,
            cand_relabel=cand_relabel,
            use_tqdm=use_tqdm,
        )
    else:   # strategy == "qa"
        return QAPromptConstructStrategy(
            prefix=prefix,
            cand_relabel=cand_relabel,
            use_tqdm=use_tqdm,
        )
