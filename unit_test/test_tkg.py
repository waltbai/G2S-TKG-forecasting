import unittest

from llm4tkg.preprocess.fact import Fact
from llm4tkg.preprocess.tkg import TemporalKG


class TestTKG(unittest.TestCase):
    """Test TemporalKG class."""

    def test_init(self):
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
            Fact("A", "R2", "C", "2024-01-04", 0, 1, 1, 3),
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
        self.assertEqual(
            tkg.statistic(),
            f"Total number of entities: 4\n"
            f"Total number of relations: 2\n"
            f"Total number of train facts: 5\n"
            f"Total number of valid facts: 4\n"
            f"Total number of test facts: 3\n\n"
        )
