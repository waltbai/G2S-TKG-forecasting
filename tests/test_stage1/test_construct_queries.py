import unittest
from datetime import datetime

from src.stage1.prepare import construct_queries
from src.utils.fact import Fact


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


class TestConstructQueries(unittest.TestCase):
    def test_construct_queries(self):
        # Case 1: train queries, single side
        train_queries = construct_queries(train_facts, False)
        self.assertEqual(len(train_queries), 5)

        # Case 2: train queries, both sides
        train_queries = construct_queries(train_facts, True)
        self.assertEqual(len(train_queries), 10)

        # Case 3: valid queries, single side
        valid_queries = construct_queries(valid_facts, False)
        self.assertEqual(len(valid_queries), 4)

        # Case 4: valid queries, both sides
        valid_queries = construct_queries(valid_facts, True)
        self.assertEqual(len(valid_queries), 8)

        # Case 5: test queries, single side
        queries = construct_queries(test_facts, False)
        self.assertEqual(len(queries), 3)

        # Case 6: test queries, both sides
        queries = construct_queries(test_facts, True)
        self.assertEqual(len(queries), 6)
