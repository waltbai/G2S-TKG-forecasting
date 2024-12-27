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
            map_strategy: str = "global",
            time_strategy: str = "global",
            map_entity: bool = True,
            map_relation: bool = True,
    ):
        # Name2id mapping
        if map_strategy == "global":
            self._mapping_global(query, tkg)
        elif map_strategy == "session":
            self._mapping_session(query, tkg)
        else:
            raise ValueError(f"Unknown map_strategy: {map_strategy}")

        # Construct prompt
        self._construct(
            query=query,
            map_entity=map_entity,
            map_relation=map_relation,
            time_strategy=time_strategy,
        )

    @staticmethod
    def _mapping_global(
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
    def _mapping_session(
            query: Query,
            tkg: TKG,
    ):
        # Count frequency
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

        # Re-map as session IDs
        ent_mapping = {ent: idx for idx, (ent, freq) in enumerate(ent_sorted)}
        if query.answer not in ent_mapping:
            ent_mapping.setdefault(query.answer, None)
        rel_mapping = {rel: idx for idx, (rel, freq) in enumerate(rel_sorted)}
        query.entity_mapping = ent_mapping
        query.rel_mapping = rel_mapping

    @staticmethod
    def _construct(
            query: Query,
            map_entity: bool = True,
            map_relation: bool = True,
            time_strategy: str = "global",
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
            # Notice: don't add answer id here! It will cause information leakage.
            ids = set()
            for fact in query.history:
                ids.add((fact.head, ent_func(fact.head)))
                ids.add((fact.tail, ent_func(fact.tail)))
            ids.add((query.entity, ent_func(query.entity)))
            ids = sorted(ids, key=lambda x: int(x[1]))
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
            if time_strategy == "global":
                fact_time = fact.time
            elif time_strategy == "session":
                fact_time = query.time - fact.time
            else:
                raise ValueError(f"Unknown time_strategy: {time_strategy}")
            prompt += (
                f"{fact_time}:["
                f"{ent_func(fact.head)}{sep}"
                f"{rel_func(fact.rel)}{sep}"
                f"{ent_func(fact.tail)}"
                f"]\n"
            )
        prompt += "\n"
        # Query part
        prompt += "### Query ###\n"
        if time_strategy == "global":
            query_time = query.time
        elif time_strategy == "session":
            query_time = 0
        else:
            raise ValueError(f"Unknown time_strategy: {time_strategy}")
        if query.role == "head":
            prompt += (
                f"{query_time}:["
                f"{ent_func(query.entity)}{sep}"
                f"{rel_func(query.rel)}{sep}"
                f"?]\n"
            )
        else:
            prompt += (
                f"{query_time}:["
                f"?{sep}"
                f"{rel_func(query.rel)}{sep}"
                f"{ent_func(query.entity)}]\n"
            )
        prompt += "\n"
        # Answer part
        prompt += "### Answer ###\n"
        query.label += f"{ent_func(query.answer)}"
        query.prompt = prompt
