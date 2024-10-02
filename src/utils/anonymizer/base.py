from abc import ABC, abstractmethod

from typing import Tuple, Dict, List

from tqdm import tqdm

from src.utils.query import Query
from src.utils.tkg import TKG


class Anonymizer(ABC):
    """Basic anonymizer class."""
    def __init__(
            self,
            anonymize_entity: bool = True,
            anonymize_rel: bool = True,
            use_tqdm: bool = False,
            **kwargs,
    ):
        self.anonymize_entity = anonymize_entity
        self.anonymize_rel = anonymize_rel
        self.use_tqdm = use_tqdm

    def __call__(
            self,
            queries: List[Query],
            tkg: TKG,
    ):
        with tqdm(total=len(queries), disable=not self.use_tqdm) as pbar:
            for query in queries:
                entity_mapping, rel_mapping = self.mapping(query, tkg)
                query.entity_mapping = entity_mapping
                query.rel_mapping = rel_mapping
                pbar.update()

    @abstractmethod
    def mapping(
            self,
            query: Query,
            tkg: TKG,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Map entities and relations to their anonymous names.

        Args:
            query (Query): Query with historical facts.
            tkg (TKG): Temporal Knowledge Graph.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]:
                entity_mapping, relation_mapping.
        """
