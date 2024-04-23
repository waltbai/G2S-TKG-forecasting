import unittest
from datetime import datetime, timedelta

from src.evaluation import Query
from src.preprocess.fact import Fact
from src.preprocess.tkg import TemporalKG
from src.prompt import quadruple_prompt

entities = ["A", "B", "C", "D"]
relations = ["R1", "R2"]
train_set = [
    Fact("A", "R1", "B", "2024-01-01"),
    Fact("A", "R2", "B", "2024-01-01"),
    Fact("A", "R1", "C", "2024-01-02"),
    Fact("C", "R2", "B", "2024-01-02"),
    Fact("C", "R1", "D", "2024-01-03"),
]
valid_set = [
    Fact("A", "R1", "B", "2024-01-04"),
    Fact("A", "R2", "C", "2024-01-04"),
    Fact("A", "R1", "B", "2024-01-05"),
    Fact("A", "R2", "B", "2024-01-05"),
]
test_set = [
    Fact("A", "R1", "B", "2024-01-06"),
    Fact("A", "R1", "C", "2024-01-06"),
    Fact("A", "R1", "D", "2024-01-07"),
]
tkg = TemporalKG(
    entities=entities,
    relations=relations,
    train_set=train_set,
    valid_set=valid_set,
    test_set=test_set,
    base_time=datetime.fromisoformat("2024-01-01"),
    time_precision="day",
    time_unit=timedelta(days=1),
    anon_entity=None,
    anon_rel=None,
    anon_time="index",
)


class TestQuadruplePrompt(unittest.TestCase):
    """Test quadruple prompt function."""

    def test_entity_uni_strings_tail(self):
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-06",
            direction="tail",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
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
        query = Query(
            entity="B",
            rel="R1",
            answers=["A"],
            time="2024-01-06",
            direction="head",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
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
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-06",
            direction="tail",
        )
        tkg.anon_entity = "index"
        tkg.anon_rel = "index"
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
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
        tkg.anon_entity = None
        tkg.anon_rel = None

    def test_pair_uni_strings(self):
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-06",
            direction="tail",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="pair",
            history_direction="uni",
        )
        self.assertEqual(
            prompt,
            f"0:[A,R1,0.B]\n"
            f"1:[A,R1,1.C]\n"
            f"3:[A,R1,0.B]\n"
            f"4:[A,R1,0.B]\n"
            f"5:[A,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "B", "1": "C"}
        )

    def test_pair_uni_strings_head(self):
        query = Query(
            entity="B",
            rel="R1",
            answers=["A"],
            time="2024-01-06",
            direction="head",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="pair",
            history_direction="uni",
        )
        self.assertEqual(
            prompt,
            f"0:[B,R1,0.A]\n"
            f"3:[B,R1,0.A]\n"
            f"4:[B,R1,0.A]\n"
            f"5:[B,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "A"}
        )

    def test_entity_bi_strings_head(self):
        query = Query(
            entity="C",
            rel="R1",
            answers=["A"],
            time="2024-01-06",
            direction="head",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="bi",
        )
        self.assertEqual(
            prompt,
            f"1:[C,R1,0.A]\n"
            f"1:[C,R2,1.B]\n"
            f"2:[C,R1,2.D]\n"
            f"3:[C,R2,0.A]\n"
            f"5:[C,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "A", "1": "B", "2": "D"}
        )

    def test_pair_bi_strings_head(self):
        query = Query(
            entity="C",
            rel="R1",
            answers=["A"],
            time="2024-01-06",
            direction="head",
        )
        prompt, candidates = quadruple_prompt(
            query=query,
            tkg=tkg,
            history_type="pair",
            history_direction="bi",
        )
        self.assertEqual(
            prompt,
            f"1:[C,R1,0.A]\n"
            f"2:[C,R1,1.D]\n"
            f"5:[C,R1,"
        )
        self.assertDictEqual(
            candidates,
            {"0": "A", "1": "D"}
        )


