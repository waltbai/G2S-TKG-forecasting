from typing import List, Dict

from src.stage1.prompt.base import PromptConstructStrategy
from src.utils.query import Query


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
            query: Query,
            sep: str = ",",
    ):
        """Main func of prompt construction."""
        query_entity, query_rel, query_answer, query_time = query_quad

        # Construct prompt
        prompt = "### History ###\n"
        for quad in his_quads:
            head, rel, tail, time = quad
            prompt += f"{time}:[{head}{sep}{rel}{sep}{tail}]\n"
        prompt += f"\n### Query ###\n"
        if query.entity_role == "head":
            prompt += f"{query_time}:[{query_entity}{sep}{query_rel}{sep}?]\n"
        else:
            prompt += f"{query_time}:[?{sep}{query_rel}{sep}{query_entity}]\n"
        prompt += f"\n### Answer ###\n"
        if query_answer is not None:
            label = query_answer
        else:
            # query_answer is never appeared in history,
            # generate a fake ID for it.
            label = str(len(query.entity_mapping))

        query.prompt = prompt
        query.label = label
        query.anonymous_filters = [
            query.entity_mapping[ent] for ent in query.filters
            if ent in query.entity_mapping
        ]
