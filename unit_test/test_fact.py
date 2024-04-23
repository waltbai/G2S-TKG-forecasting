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
