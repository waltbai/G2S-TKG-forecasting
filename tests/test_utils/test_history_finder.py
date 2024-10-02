import unittest
from datetime import datetime

from src.stage1.prompt import get_prompt_constructor
from src.utils.anonymizer import get_anonymizer
from src.utils.fact import Fact
from src.utils.history_finder import get_history_finder
from src.utils.query import construct_queries
from src.utils.time_processor import get_time_processor
from src.utils.tkg import construct_search_histories, TKG

entities = ["A", "B", "C", "D"]
entity2id = {"A": 0, "B": 1, "C": 2, "D": 3}
relations = ["R1", "R2"]
relation2id = {"R1": 0, "R2": 1}
train_facts = [
    Fact("A", "R1", "B", "2024-01-01"),
    Fact("A", "R2", "B", "2024-01-01"),
    Fact("A", "R1", "C", "2024-01-02"),
    Fact("C", "R2", "B", "2024-01-02"),
    Fact("C", "R1", "D", "2024-01-03"),
]
valid_facts = [
    Fact("A", "R1", "B", "2024-01-04"),
    Fact("A", "R2", "C", "2024-01-04"),
    Fact("A", "R1", "B", "2024-01-05"),
    Fact("A", "R2", "B", "2024-01-05"),
]
test_facts = [
    Fact("A", "R1", "B", "2024-01-06"),
    Fact("A", "R1", "C", "2024-01-06"),
    Fact("A", "R1", "D", "2024-01-07"),
]
base_time = datetime.fromisoformat("2024-01-01")
facts = train_facts + valid_facts + test_facts
search_history = construct_search_histories(facts)
time2id = {
    "2024-01-01": 0,
    "2024-01-02": 1,
    "2024-01-03": 2,
    "2024-01-04": 3,
    "2024-01-05": 4,
    "2024-01-06": 5,
    "2024-01-07": 6,
}
id2time = {v: k for k, v in time2id.items()}
tkg = TKG(
    name="test",
    train_facts=train_facts,
    valid_facts=valid_facts,
    test_facts=test_facts,
    base_time=base_time,
    time_unit="day",
    entities=entities,
    entity2id=entity2id,
    relations=relations,
    relation2id=relation2id,
    search_history=search_history,
    time2id=time2id,
    id2time=id2time,
)


class TestQAPromptConstructor(unittest.TestCase):
    def test_session_query(self):
        history_finder = get_history_finder(
            history_finder="2-hop",
            history_type="entity",
            history_direction="uni",
        )
        anonymizer = get_anonymizer(
            strategy="session",
            anonymize_entity=True,
            anonymize_rel=True,
            use_tqdm=False,
        )
        time_processor = get_time_processor(
            strategy="query",
            use_tqdm=False,
        )
        prompt_constructor = get_prompt_constructor(
            strategy="qa",
            use_tqdm=False,
        )
        queries = construct_queries(test_facts)
        history_finder(queries, tkg)
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        # Case 1: predict tail
        query = queries[0]
        expected_prompt = \
            "### History ###\n" \
            "5:[0,0,1]\n" \
            "5:[0,1,1]\n" \
            "4:[0,0,2]\n" \
            "2:[0,0,1]\n" \
            "2:[0,1,2]\n" \
            "1:[0,0,1]\n" \
            "1:[0,1,1]\n\n" \
            "### Query ###\n" \
            "0:[0,0,?]\n\n" \
            "### Answer ###\n"
        expected_label = "1"
        expected_filters = ["2"]
        expected_candidates = {"0": "A", "1": "B", "2": "C", "3": "D"}
        self.assertEqual(query.prompt, expected_prompt)
        self.assertEqual(query.label, expected_label)
        self.assertListEqual(query.anonymous_filters, expected_filters)

