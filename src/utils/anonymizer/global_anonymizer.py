from typing import Tuple, Dict

from src.utils.anonymizer.base import Anonymizer
from src.utils.query import Query
from src.utils.tkg import TKG


class GlobalAnonymizer(Anonymizer):
    """Global anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Define anonymize function
        anonymize_ent = lambda entity: entity if not self.anonymize_entity \
            else str(tkg.entity2id[entity])
        anonymize_rel = lambda rel: rel if not self.anonymize_rel \
            else str(tkg.relation2id[rel])

        # Construct mappings
        entity_mapping = {}
        rel_mapping = {}
        if self.anonymize_entity:
            entity_mapping.setdefault(query.entity, anonymize_ent(query.entity))
        else:
            entity_mapping.setdefault(query.entity, query.entity)
        entity_mapping.setdefault(query.answer, anonymize_ent(query.answer))
        rel_mapping.setdefault(query.rel, anonymize_rel(query.rel))
        for fact in query.history:
            entity_mapping.setdefault(fact.head, anonymize_ent(fact.head))
            entity_mapping.setdefault(fact.tail, anonymize_ent(fact.tail))
            rel_mapping.setdefault(fact.rel, anonymize_rel(fact.rel))

        return entity_mapping, rel_mapping
