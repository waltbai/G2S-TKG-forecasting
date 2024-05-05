import unittest

from src.utils.fact import Fact


class TestFact(unittest.TestCase):
    """Test Fact class."""

    def test_create_from_ids(self):
        params = {
            "entities": ["A", "B", "C", "D"],
            "relations": ["R1", "R2"],
            "id2time": {
                0: "2024-01-01",
                1: "2024-01-02",
                2: "2024-01-03",
            }
        }
        # Case 1: basic
        fact = Fact.from_ids([0, 0, 1, 0], **params)
        self.assertEqual(fact, Fact("A", "R1", "B", "2024-01-01"))
        # Case 2: change time ids
        fact = Fact.from_ids([2, 1, 3, 2], **params)
        self.assertEqual(fact, Fact("C", "R2", "D", "2024-01-03"))

    def test_fact_quadruple(self):
        fact = Fact("S", "V", "O", "T")
        # Case 1: default param: "normal"
        self.assertEqual(fact.quadruple(), ("S", "V", "O", "T"))
        # Case 2: normal quadruple
        self.assertEqual(fact.quadruple("normal"), ("S", "V", "O", "T"))
        # Case 3: swap head and tail
        self.assertEqual(fact.quadruple("swap"), ("O", "V", "S", "T"))
        # Case 4: inverse the relation
        self.assertEqual(fact.quadruple("inverse"), ("O", "inverse V", "S", "T"))
