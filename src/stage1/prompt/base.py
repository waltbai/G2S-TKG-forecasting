from abc import ABC, abstractmethod

from typing import List, Dict

from tqdm import tqdm

from src.utils.query import Query


class PromptConstructStrategy(ABC):
    """Prompt construct strategy base class."""

    def __init__(
            self,
            use_tqdm: bool = False
    ):
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
            his_quads.append(quad)

        # Construct prompt
        self.construct_prompt(
            query_quad=query_quad,
            his_quads=his_quads,
            query=query,
            sep=sep,
        )
