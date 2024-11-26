import unittest

from src.utils.data.fact import Fact
from src.utils.data.query import Query
from src.utils.data.tkg import TKG


class TestTKG(unittest.TestCase):
    """
    Unit test for TKG class.
    """

    def test_load(self):
        tkg = TKG.load("tests", "tkg")

        # Check train/valid/test fact number
        self.assertEqual(len(tkg.train_facts), 5)
        self.assertEqual(len(tkg.valid_facts), 4)
        self.assertEqual(len(tkg.test_facts), 3)

        # Check dict
        self.assertDictEqual(
            tkg.entity2id,
            {"A": "0", "B": "1", "C": "2", "D": "3"}
        )
        self.assertDictEqual(
            tkg.relation2id,
            {"R1": "0", "R2": "1"}
        )

        # Check rules
        self.assertDictEqual(tkg.rules, {})

        # Check indices
        self.assertEqual(len(tkg.index["head"]), 2)
        self.assertEqual(len(tkg.index["tail"]), 3)
        self.assertEqual(len(tkg.index["head+rel"]), 4)
        self.assertEqual(len(tkg.index["tail+rel"]), 5)

        # Check fact
        self.assertEqual(tkg.train_facts[0], Fact("A", "R1", "B", 0))

    def test_construct_queries(self):
        tkg = TKG.load("tests", "tkg")
        train_queries = tkg.construct_queries("train")
        valid_queries = tkg.construct_queries("valid")
        test_queries = tkg.construct_queries("test")

        # Check query number
        self.assertEqual(len(train_queries), 10)
        self.assertEqual(len(valid_queries), 8)
        self.assertEqual(len(test_queries), 6)

    def test_find_history(self):
        tkg = TKG.load("tests", "tkg")
        query = Query(
            entity="A",
            rel="R1",
            answer="B",
            time=5,
            role="head"
        )

        # Check find history by hop
        tkg.find_history(query, "hop")
        self.assertListEqual(
            query.history,
            [
                Fact("A", "R1", "B", 0),
                Fact("A", "R2", "B", 0),
                Fact("A", "R1", "C", 1),
                Fact("A", "R1", "B", 3),
                Fact("A", "R2", "C", 3),
                Fact("A", "R1", "B", 4),
                Fact("A", "R2", "B", 4),
            ]
        )

        # Check find history by rule
        rules = {"R1": {"R1": 0.5}}
        tkg.rules = rules
        tkg.find_history(query, "rule")
        self.assertListEqual(
            query.history,
            [
                Fact("A", "R1", "B", 0),
                Fact("A", "R1", "C", 1),
                Fact("A", "R1", "B", 3),
                Fact("A", "R1", "B", 4),
            ]
        )
