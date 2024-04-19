import unittest

from src.preprocess.fact import Fact


class TestFact(unittest.TestCase):
    """Test Fact class."""

    def test_fact_quadruple(self):
        fact = Fact("S", "V", "O", "T")
        self.assertEqual(fact.quadruple(), ("S", "V", "O", "T"))
        self.assertEqual(fact.quadruple("normal"), ("S", "V", "O", "T"))
        self.assertEqual(fact.quadruple("swap"), ("O", "V", "S", "T"))
        self.assertEqual(fact.quadruple("inverse"), ("O", "inverse V", "S", "T"))

    def test_fact_prompt_quadruple(self):
        fact = Fact("S", "V", "O", "T", 0, 1, 2, 3)
        self.assertEqual(fact.prompt_quadruple(
            query_entity="S",
            anonymize=False,
            anonymize_time=False,
        ), ("S", "V", "O", "T"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="S",
            anonymize=False,
            anonymize_time=True,
        ), ("S", "V", "O", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="S",
            anonymize=True,
            anonymize_time=True,
        ), ("0", "1", "2", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymize=False,
            anonymize_time=False,
        ), ("O", "V", "S", "T"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymize=False,
            anonymize_time=True,
        ), ("O", "V", "S", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymize=True,
            anonymize_time=True,
        ), ("2", "1", "0", "3"))
