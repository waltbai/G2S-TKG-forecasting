import unittest
from datetime import datetime

from src.stage1.prepare import construct_queries, construct_history
from src.utils.time_processor import get_time_processor
from src.utils.fact import Fact
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


class TestTimeProcessor(unittest.TestCase):
    def test_absolute_time_process_strategy(self):
        time_processor = get_time_processor("absolute")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1
        time_processor(queries, tkg)
        self.assertDictEqual(
            queries[0].time_mapping,
            {
                "2024-01-01": "2024-01-01",
                "2024-01-02": "2024-01-02",
                "2024-01-04": "2024-01-04",
                "2024-01-05": "2024-01-05",
                "2024-01-06": "2024-01-06",
             }
        )

    def test_start_time_process_strategy(self):
        time_processor = get_time_processor("start")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1
        time_processor(queries, tkg)
        self.assertDictEqual(
            queries[0].time_mapping,
            {
                "2024-01-01": "0",
                "2024-01-02": "1",
                "2024-01-04": "3",
                "2024-01-05": "4",
                "2024-01-06": "5",
            }
        )

    def test_query_time_process_strategy(self):
        time_processor = get_time_processor("query")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1
        time_processor(queries, tkg)
        self.assertDictEqual(
            queries[0].time_mapping,
            {
                "2024-01-01": "5",
                "2024-01-02": "4",
                "2024-01-04": "2",
                "2024-01-05": "1",
                "2024-01-06": "0",
            }
        )
