import unittest
from datetime import datetime

from src.utils.fact import Fact
from src.utils.tkg import TKG, construct_search_histories

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


class TestSearchHistory(unittest.TestCase):
    """Test search history function."""

    def test_search_histories_head(self):
        # Test head
        self.assertListEqual(
            search_history["head"]["A"],
            [
                Fact("A", "R1", "B", "2024-01-01"),
                Fact("A", "R2", "B", "2024-01-01"),
                Fact("A", "R1", "C", "2024-01-02"),
                Fact("A", "R1", "B", "2024-01-04"),
                Fact("A", "R2", "C", "2024-01-04"),
                Fact("A", "R1", "B", "2024-01-05"),
                Fact("A", "R2", "B", "2024-01-05"),
                Fact("A", "R1", "B", "2024-01-06"),
                Fact("A", "R1", "C", "2024-01-06"),
                Fact("A", "R1", "D", "2024-01-07"),
            ]
        )

    def test_search_histories_tail(self):
        # Test tail
        self.assertListEqual(
            search_history["tail"]["B"],
            [
                Fact("A", "R1", "B", "2024-01-01"),
                Fact("A", "R2", "B", "2024-01-01"),
                Fact("C", "R2", "B", "2024-01-02"),
                Fact("A", "R1", "B", "2024-01-04"),
                Fact("A", "R1", "B", "2024-01-05"),
                Fact("A", "R2", "B", "2024-01-05"),
                Fact("A", "R1", "B", "2024-01-06"),
            ]
        )

    def test_search_histories_both(self):
        self.assertListEqual(
            search_history["both"]["C"],
            [
                Fact("A", "R1", "C", "2024-01-02"),
                Fact("C", "R2", "B", "2024-01-02"),
                Fact("C", "R1", "D", "2024-01-03"),
                Fact("A", "R2", "C", "2024-01-04"),
                Fact("A", "R1", "C", "2024-01-06"),
            ]
        )

    def test_search_histories_head_rel(self):
        self.assertEqual(
            search_history["head_rel"][("A", "R2")],
            [
                Fact("A", "R2", "B", "2024-01-01"),
                Fact("A", "R2", "C", "2024-01-04"),
                Fact("A", "R2", "B", "2024-01-05"),
            ]
        )

    def test_search_histories_tail_rel(self):
        self.assertListEqual(
            search_history["tail_rel"][("B", "R2")],
            [
                Fact("A", "R2", "B", "2024-01-01"),
                Fact("C", "R2", "B", "2024-01-02"),
                Fact("A", "R2", "B", "2024-01-05"),
            ]
        )

    def test_search_histories_both_rel(self):
        self.assertListEqual(
            search_history["both_rel"][("C", "R1")],
            [
                Fact("A", "R1", "C", "2024-01-02"),
                Fact("C", "R1", "D", "2024-01-03"),
                Fact("A", "R1", "C", "2024-01-06"),
            ]
        )


class TestTKG(unittest.TestCase):
    def test_init(self):
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
        self.assertEqual(len(tkg.train_facts), 5)
        self.assertEqual(len(tkg.valid_facts), 4)
        self.assertEqual(len(tkg.test_facts), 3)
