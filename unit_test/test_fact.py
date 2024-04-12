import unittest

from llm4tkg.preprocess.fact import Fact


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
            anonymous=False,
            anonymous_time=False,
        ), ("S", "V", "O", "T"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="S",
            anonymous=False,
            anonymous_time=True,
        ), ("S", "V", "O", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="S",
            anonymous=True,
            anonymous_time=True,
        ), ("0", "1", "2", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymous=False,
            anonymous_time=False,
        ), ("O", "V", "S", "T"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymous=False,
            anonymous_time=True,
        ), ("O", "V", "S", "3"))
        self.assertEqual(fact.prompt_quadruple(
            query_entity="O",
            anonymous=True,
            anonymous_time=True,
        ), ("2", "1", "0", "3"))
