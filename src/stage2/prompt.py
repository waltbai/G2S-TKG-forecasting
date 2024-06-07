from abc import ABC, abstractmethod

from typing import List, Dict

from tqdm import tqdm

from src.utils.query import Query


class PromptConstructStrategy(ABC):
    """Prompt construct strategy base class."""

    def __init__(
            self,
            deanonymize_strategy: str = "fillin",
            use_tqdm: bool = False
    ):
        self.deanonymize_strategy = deanonymize_strategy
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
        query_quad, his_quads, cand_mapping = self.prepare_elements(query)
        self.construct_prompt(
            query_quad=query_quad,
            his_quads=his_quads,
            cand_mapping=cand_mapping,
            query=query,
            sep=sep,
        )

    def prepare_elements(self, query: Query):
        """Prepare elements in prompt."""
        if self.deanonymize_strategy == "fillin":
            query_entity = query.entity
            query_rel = query.rel
            query_answer = query.answer
            query_time = query.time
            map_quad = lambda x: [
                x.head,
                x.rel,
                x.tail,
                x.time,
            ]
        else:
            query_entity = query.entity_mapping[query.entity]
            query_rel = query.rel_mapping[query.rel]
            query_answer = query.entity_mapping[query.answer] \
                if query.answer in query.entity_mapping else None
            query_time = query.time_mapping[query.time]
            map_quad = lambda x: [
                query.entity_mapping[x.head],
                query.rel_mapping[x.rel],
                query.entity_mapping[x.tail],
                query.time_mapping[x.time]
            ]
        query_quad = [query_entity, query_rel, query_answer, query_time]

        # Convert historical facts into quadruples
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

        return query_quad, his_quads, cand_mapping


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
        query_entity, query_rel, query_answer, query_time = query_quad

        # Construct prompt: De-anonymized part
        prompt = ""
        if self.deanonymize_strategy == "map":
            prompt += "### Entity Mapping ###\n"
            ent_maps = list(
                sorted(query.entity_mapping.items(), key=lambda x: x[1])
            )
            for k, v in ent_maps:
                prompt += f"{v}:{k}\n"
            prompt += "\n"
            prompt += "### Relation Mapping ###\n"
            rel_maps = list(
                sorted(query.rel_mapping.items(), key=lambda x: x[1])
            )
            for k, v in rel_maps:
                prompt += f"{v}:{k}\n"
            prompt += "\n"

        # Construct prompt: History and query part
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

    def construct_prompt(
            self,
            query_quad: List[str],
            his_quads: List[List[str]],
            cand_mapping: Dict[str, str],
            query: Query,
            sep: str = ",",
    ):
        query_entity, query_rel, query_answer, query_time = query_quad

        # Construct prompt: De-anonymized part
        prompt = ""
        if self.deanonymize_strategy == "map":
            prompt += "### Entity Mapping ###\n"
            ent_maps = list(
                sorted(query.entity_mapping.items(), key=lambda x: x[1])
            )
            for k, v in ent_maps:
                prompt += f"{v}:{k}\n"
            prompt += "\n"
            prompt += "### Relation Mapping ###\n"
            rel_maps = list(
                sorted(query.rel_mapping.items(), key=lambda x: x[1])
            )
            for k, v in rel_maps:
                prompt += f"{v}:{k}\n"
            prompt += "\n"

        # Construct prompt: History and query part
        prompt += "### History ###\n"
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
        prompt_construct_strategy: str,
        deanonymize_strategy: str = "fillin",
        use_tqdm: bool = False,
) -> PromptConstructStrategy:
    """Get prompt constructor by name."""
    if prompt_construct_strategy == "inline":
        return InlinePromptConstructStrategy(
            deanonymize_strategy, use_tqdm
        )
    else:   # strategy == "qa"
        return QAPromptConstructStrategy(
            deanonymize_strategy,
            use_tqdm
        )
