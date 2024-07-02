"""We implement two anonymize strategies, namely, global and session (order/random).

Global strategy use tkg ID as the anonymous name for elements.

Session order strategy use frequency to order and re-label IDs
    with-in a session (query and its history) as the anonymous name for elements.

Session random strategy randomly re-label IDs with-in a session
    as the anonymous name for elements.
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

from tqdm import tqdm

from src.utils.query import Query
from src.utils.tkg import TKG


class AnonymizeStrategy(ABC):
    """Basic class for anonymize strategies."""
    def __init__(self, use_tqdm: bool = False):
        self.use_tqdm = use_tqdm

    @abstractmethod
    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Map entities and relations to their anonymous names.

        Args:
            query (AnonymizedQuery): Anonymized query.
            tkg (TKG): Temporal Knowledge Graph.
            prefix (bool):
                Whether to apply prefix on triple elements.
                Defaults to False.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]:
                entity_mapping, relation_mapping.
        """

    def __call__(
            self,
            queries: List[Query],
            tkg: TKG,
            prefix: bool = False,
    ):
        with tqdm(total=len(queries), disable=not self.use_tqdm) as pbar:
            for query in queries:
                entity_mapping, rel_mapping = self.mapping(query, tkg, prefix)
                query.entity_mapping = entity_mapping
                query.rel_mapping = rel_mapping
                pbar.update()


class GlobalAnonymizeStrategy(AnonymizeStrategy):
    """Global anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Define anonymize function
        anonymize_ent = lambda x: f"ENT_{tkg.entity2id[x]}" if prefix \
            else str(tkg.entity2id[x])
        anonymize_rel = lambda x: f"REL_{tkg.relation2id[x]}" if prefix \
            else str(tkg.relation2id[x])

        # Construct mappings
        entity_mapping = {}
        rel_mapping = {}
        entity_mapping.setdefault(query.entity, anonymize_ent(query.entity))
        entity_mapping.setdefault(query.answer, anonymize_ent(query.answer))
        rel_mapping.setdefault(query.rel, anonymize_rel(query.rel))
        for fact in query.history:
            entity_mapping.setdefault(fact.head, anonymize_ent(fact.head))
            entity_mapping.setdefault(fact.tail, anonymize_ent(fact.tail))
            rel_mapping.setdefault(fact.rel, anonymize_rel(fact.rel))

        return entity_mapping, rel_mapping


class SessionOrderAnonymizeStrategy(AnonymizeStrategy):
    """Session order anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Count frequency of entities and relations and sort
        ent_freq = {query.entity: 1}
        rel_freq = {query.rel: 1}
        for fact in query.history:
            ent_freq.setdefault(fact.head, 0)
            ent_freq[fact.head] += 1
            ent_freq.setdefault(fact.tail, 0)
            ent_freq[fact.tail] += 1
            rel_freq.setdefault(fact.rel, 0)
            rel_freq[fact.rel] += 1
        ent_sorted = list(sorted(ent_freq.items(), key=lambda x: x[1], reverse=True))
        rel_sorted = list(sorted(rel_freq.items(), key=lambda x: x[1], reverse=True))

        # Re-map entities to session IDs
        ents = [k for k, v in ent_sorted]
        entity_mapping = {}
        for idx, ent in enumerate(ents):
            if prefix:
                entity_mapping.setdefault(ent, f"ENT_{idx}")
            else:
                entity_mapping.setdefault(ent, str(idx))

        # Re-map relations to session IDs
        rels = [k for k, v in rel_sorted]
        rel_mapping = {}
        for idx, rel in enumerate(rels):
            if prefix:
               rel_mapping.setdefault(rel, f"REL_{idx}")
            else:
               rel_mapping.setdefault(rel, str(idx))
            # rel_mapping.setdefault(rel, rel)

        return entity_mapping, rel_mapping


class SessionOrderEntityOnlyAnonymizeStrategy(AnonymizeStrategy):
    """Session order anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Count frequency of entities and relations and sort
        ent_freq = {query.entity: 1}
        rel_freq = {query.rel: 1}
        for fact in query.history:
            ent_freq.setdefault(fact.head, 0)
            ent_freq[fact.head] += 1
            ent_freq.setdefault(fact.tail, 0)
            ent_freq[fact.tail] += 1
            rel_freq.setdefault(fact.rel, 0)
            rel_freq[fact.rel] += 1
        ent_sorted = list(sorted(ent_freq.items(), key=lambda x: x[1], reverse=True))
        rel_sorted = list(sorted(rel_freq.items(), key=lambda x: x[1], reverse=True))

        # Re-map entities to session IDs
        ents = [k for k, v in ent_sorted]
        entity_mapping = {}
        for idx, ent in enumerate(ents):
            if prefix:
                entity_mapping.setdefault(ent, f"ENT_{idx}")
            else:
                entity_mapping.setdefault(ent, str(idx))

        # Re-map relations to session IDs
        rels = [k for k, v in rel_sorted]
        rel_mapping = {}
        for idx, rel in enumerate(rels):
            rel_mapping.setdefault(rel, rel)

        return entity_mapping, rel_mapping


class SessionRandomAnonymizeStrategy(AnonymizeStrategy):
    """Session random anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Count frequency of entities and relations and sort
        ents = {query.entity}
        rels = {query.rel}
        for fact in query.history:
            ents.add(fact.head)
            ents.add(fact.tail)
            rels.add(fact.rel)

        # Re-map entities to session IDs
        entity_mapping = {}
        ents = list(sorted(ents))
        random.shuffle(ents)
        for idx, ent in enumerate(ents):
            if prefix:
                entity_mapping.setdefault(ent, f"ENT_{idx}")
            else:
                entity_mapping.setdefault(ent, str(idx))

        # Re-map relations to session IDs
        rel_mapping = {}
        rels = list(sorted(rels))
        random.shuffle(rels)
        for idx, rel in enumerate(rels):
            if prefix:
                rel_mapping.setdefault(rel, f"REL_{idx}")
            else:
                rel_mapping.setdefault(rel, str(idx))

        return entity_mapping, rel_mapping


class OriginAnonymizeStrategy(AnonymizeStrategy):
    """Do not conduct anonymize."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
            prefix: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        entity_mapping = {query.entity: query.entity}
        rel_mapping = {query.rel: query.rel}

        for fact in query.history:
            entity_mapping.setdefault(fact.head, fact.head)
            entity_mapping.setdefault(fact.tail, fact.tail)
            rel_mapping.setdefault(fact.rel, fact.rel)

        return entity_mapping, rel_mapping


def get_anonymizer(
        strategy: str,
        use_tqdm: bool = False,
) -> AnonymizeStrategy:
    """Get anonymizer by strategy name."""
    if strategy == "global":
        return GlobalAnonymizeStrategy(use_tqdm)
    elif strategy == "session":
        return SessionOrderAnonymizeStrategy(use_tqdm)
    elif strategy == "session-ent":
        return SessionOrderEntityOnlyAnonymizeStrategy(use_tqdm)
    elif strategy == "random":
        return SessionRandomAnonymizeStrategy(use_tqdm)
    else:   # strategy == "origin"
        return OriginAnonymizeStrategy(use_tqdm)
