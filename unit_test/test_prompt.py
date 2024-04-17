import unittest

from llm4tkg.preprocess.fact import Fact
from llm4tkg.preprocess.tkg import TemporalKG
from llm4tkg.prompt import quadruple_prompt

entities = ["A", "B", "C", "D"]
relations = ["R1", "R2"]
train_set = [
    Fact("A", "R1", "B", "2024-01-01", 0, 0, 1, 0),
    Fact("A", "R2", "B", "2024-01-01", 0, 1, 1, 0),
    Fact("A", "R1", "C", "2024-01-02", 0, 0, 2, 1),
    Fact("C", "R2", "B", "2024-01-02", 2, 1, 1, 1),
    Fact("C", "R1", "D", "2024-01-03", 2, 0, 3, 2),
]
valid_set = [
    Fact("A", "R1", "B", "2024-01-04", 0, 0, 1, 3),
    Fact("A", "R2", "C", "2024-01-04", 0, 1, 2, 3),
    Fact("A", "R1", "B", "2024-01-05", 0, 0, 1, 4),
    Fact("A", "R2", "B", "2024-01-05", 0, 1, 1, 4),
]
test_set = [
    Fact("A", "R1", "B", "2024-01-06", 0, 0, 1, 5),
    Fact("A", "R1", "C", "2024-01-06", 0, 0, 2, 5),
    Fact("A", "R1", "D", "2024-01-07", 0, 0, 3, 6),
]
tkg = TemporalKG(
    entities=entities,
    relations=relations,
    train_set=train_set,
    valid_set=valid_set,
    test_set=test_set,
    base_time="2024-01-01",
    time_unit="day",
)


class TestQuadruplePrompt(unittest.TestCase):
    """Test quadruple prompt function."""

    def test_entity_uni_strings_tail(self):
        query = Fact("A", "R1", "B", "2024-01-06", 0, 0, 1, 5)
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
            anonymize=False,
            anonymize_time=True,
            query_target="tail",
        )
        self.assertEqual(
            prompt,
            f"0:[A,R1,0.B]\n"
            f"0:[A,R2,0.B]\n"
            f"1:[A,R1,1.C]\n"
            f"3:[A,R1,0.B]\n"
            f"3:[A,R2,1.C]\n"
            f"4:[A,R1,0.B]\n"
            f"4:[A,R2,0.B]\n"
            f"5:[A,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "B", "1": "C"}
        )

    def test_entity_uni_strings_head(self):
        query = Fact("A", "R1", "B", "2024-01-06", 0, 0, 1, 5)
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
            anonymize=False,
            anonymize_time=True,
            query_target="head",
        )
        self.assertEqual(
            prompt,
            f"0:[B,R1,0.A]\n"
            f"0:[B,R2,0.A]\n"
            f"1:[B,R2,1.C]\n"
            f"3:[B,R1,0.A]\n"
            f"4:[B,R1,0.A]\n"
            f"4:[B,R2,0.A]\n"
            f"5:[B,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "A", "1": "C"}
        )

    def test_entity_uni_indices(self):
        query = Fact("A", "R1", "B", "2024-01-06", 0, 0, 1, 5)
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
            anonymize=True,
            anonymize_time=True,
            query_target="tail",
        )
        self.assertEqual(
            prompt,
            f"0:[0,0,0.1]\n"
            f"0:[0,1,0.1]\n"
            f"1:[0,0,1.2]\n"
            f"3:[0,0,0.1]\n"
            f"3:[0,1,1.2]\n"
            f"4:[0,0,0.1]\n"
            f"4:[0,1,0.1]\n"
            f"5:[0,0,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "1", "1": "2"}
        )

    def test_pair_uni_strings(self):
        pass

    def test_pair_bi_strings(self):
        pass

