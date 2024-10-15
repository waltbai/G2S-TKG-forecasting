from typing import Tuple, Dict

from src.utils.data.query import Query
from src.utils.data.tkg import TKG


class PromptConstructor:
    """
    Prompt constructor class.
    """

    def __call__(
            self,
            query: Query,
            tkg: TKG,
            map_entity: bool = True,
            map_relation: bool = True,
    ):
        self._mapping(query, tkg)
        self._construct(query, map_entity, map_relation)

    @staticmethod
    def _mapping(
            query: Query,
            tkg: TKG,
    ):
        """Construct {name: id} mapping for entities and relations."""
        entity_mapping = {
            query.entity: tkg.entity2id[query.entity],
            query.answer: tkg.entity2id[query.answer]
        }
        rel_mapping = {
            query.rel: tkg.relation2id[query.rel]
        }
        for fact in query.history:
            entity_mapping.setdefault(fact.head, tkg.entity2id[fact.head])
            entity_mapping.setdefault(fact.tail, tkg.entity2id[fact.tail])
            rel_mapping.setdefault(fact.rel, tkg.relation2id[fact.rel])
        query.entity_mapping = entity_mapping
        query.rel_mapping = rel_mapping

    @staticmethod
    def _construct(
            query: Query,
            map_entity: bool = True,
            map_relation: bool = True,
    ):
        """
        Construct prompt.

        The simplest one, using pre-defined time stamp, entity ID and rel ID.
        """
        prompt = ""
        sep = ","
        ent_func = lambda x: query.entity_mapping[x]
        rel_func = lambda x: query.rel_mapping[x]

        # Entity mapping part
        if map_entity:
            prompt += "### Entities ###\n"
            ids = sorted(query.entity_mapping.items(), key=lambda x:int(x[1]))
            for word, i in ids:
                prompt += f"{i}:{word}\n"
            prompt += "\n"
        else:
            pass
        if map_relation:
            prompt += "### Relations ###\n"
            ids = sorted(query.rel_mapping.items(), key=lambda x:int(x[1]))
            for word, i in ids:
                prompt += f"{i}:{word}\n"
            prompt += "\n"
        else:
            pass
        # History part
        prompt += "### History ###\n"
        for fact in query.history:
            prompt += (
                f"{fact.time}:["
                f"{ent_func(fact.head)}{sep}"
                f"{rel_func(fact.rel)}{sep}"
                f"{ent_func(fact.tail)}"
                f"]\n"
            )
        prompt += "\n"
        # Query part
        prompt += "### Query ###\n"
        if query.role == "head":
            prompt += (
                f"{query.time}:["
                f"{ent_func(query.entity)}{sep}"
                f"{rel_func(query.rel)}{sep}"
                f"?]\n"
            )
        else:
            prompt += (
                f"{query.time}:["
                f"?{sep}"
                f"{rel_func(query.rel)}{sep}"
                f"{ent_func(query.entity)}]\n"
            )
        prompt += "\n"
        # Answer part
        prompt += "### Answer ###\n"
        label = f"{ent_func(query.answer)}"
        query.prompt = prompt
        query.label = label
