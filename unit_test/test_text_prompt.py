import unittest

from src.evaluation import Query
from src.preprocess.fact import Fact
from src.preprocess.tkg import TemporalKG
from src.prompt import natural_language_prompt

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
    Fact("A", "R1", "B", "2024-01-12", 0, 0, 1, 11),
    Fact("A", "R1", "C", "2024-01-12", 0, 0, 2, 11),
    Fact("A", "R1", "D", "2024-01-13", 0, 0, 3, 12),
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


class TestTextPrompt(unittest.TestCase):
    """Test text prompt function."""

    def test_entity_uni_tail(self):
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-12",
            entity_idx=0,
            rel_idx=0,
            answers_idx=[1, 2],
            time_idx=11,
            direction="tail",
        )
        prompt, candidates = natural_language_prompt(
            query=query,
            tkg=tkg,
            history_type="entity",
            history_direction="uni",
            anonymize=False,
            anonymize_time=True,
        )
        self.assertEqual(
            prompt,
            f"A,R1,B,on the 0th day;\n"
            f"A,R2,B,on the 0th day;\n"
            f"A,R1,C,on the 1st day;\n"
            f"A,R1,B,on the 3rd day;\n"
            f"A,R2,C,on the 3rd day;\n"
            f"A,R1,B,on the 4th day;\n"
            f"A,R2,B,on the 4th day;\n"
            f"Here is the query:\n"
            f"A,R1,whom,on the 11th day?"
        )
        self.assertDictEqual(
            candidates,
            {"0": "B", "1": "C"}
        )
