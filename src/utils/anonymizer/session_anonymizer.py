from typing import Tuple, Dict

from src.utils.anonymizer.base import Anonymizer
from src.utils.query import Query
from src.utils.tkg import TKG


class SessionAnonymizer(Anonymizer):
    """Session order anonymize strategy class."""

    def mapping(
            self,
            query: Query,
            tkg: TKG,
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

        # Re-label session IDs to entities and relations
        anonymize_ent = lambda idx, ent: ent if not self.anonymize_entity \
            else str(idx)
        anonymize_rel = lambda idx, rel: rel if not self.anonymize_rel \
            else str(idx)

        ents = [k for k, v in ent_sorted]
        entity_mapping = {}
        for idx, ent in enumerate(ents):
            entity_mapping.setdefault(ent, anonymize_ent(idx, ent))

        rels = [k for k, v in rel_sorted]
        rel_mapping = {}
        for idx, rel in enumerate(rels):
            rel_mapping.setdefault(rel, anonymize_rel(idx, rel))

        return entity_mapping, rel_mapping
