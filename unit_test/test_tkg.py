import unittest
from datetime import datetime, timedelta

from src.preprocess.fact import Fact
from src.preprocess.tkg import TemporalKG


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
    time_unit=timedelta(days=1),
    time_precision="day",
    valid_queries=True,
    anon_entity=None,
    anon_rel=None,
    anon_time=None,
)


class TestTKG(unittest.TestCase):
    """Test TemporalKG class."""

    def test_init(self):
        """Check if tkg initializes well."""
        self.assertEqual(
            tkg.statistic(),
            f"# entities      : 4\n"
            f"# relations     : 2\n"
            f"# train facts   : 5\n"
            f"# valid facts   : 4\n"
            f"# test facts    : 3\n"
            f"# valid queries : 8\n"
            f"# test queries  : 5\n"
        )

    def test_head_index(self):
        """Check if tkg can correctly get history by head (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_head("A")
        self.assertListEqual(
            his1,
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
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_head("A")
        self.assertListEqual(his1, his2)

    def test_head_index_2(self):
        """Check if tkg can correctly get history by head (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_head("C")
        self.assertListEqual(
            his1,
            [
                Fact("C", "R2", "B", "2024-01-02"),
                Fact("C", "R1", "D", "2024-01-03"),
            ]
        )
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_head("C")
        self.assertListEqual(his1, his2)

    def test_tail_index(self):
        """Check if tkg can correctly get history by tail (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_tail("B")
        self.assertListEqual(
            his1,
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
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_tail("B")
        self.assertListEqual(his1, his2)

    def test_both_index(self):
        """Check if tkg can correctly get history by entity (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_both("C")
        self.assertListEqual(
            his1,
            [
                Fact("A", "R1", "C", "2024-01-02"),
                Fact("C", "R2", "B", "2024-01-02"),
                Fact("C", "R1", "D", "2024-01-03"),
                Fact("A", "R2", "C", "2024-01-04"),
                Fact("A", "R1", "C", "2024-01-06"),
            ]
        )
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_both("C")
        self.assertListEqual(his1, his2)

    def test_head_rel_index(self):
        """Check if tkg can correctly get history by
        head and relation (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_head_rel("A", "R2")
        self.assertListEqual(
            his1,
            [
                Fact("A", "R2", "B", "2024-01-01"),
                Fact("A", "R2", "C", "2024-01-04"),
                Fact("A", "R2", "B", "2024-01-05"),
            ]
        )
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_head_rel("A", "R2")
        self.assertListEqual(his1, his2)

    def test_tail_rel_index(self):
        """Check if tkg can correctly get history by
        tail and relation (and via indices)."""
        tkg.clear_indices()
        his1 = tkg.find_history_by_tail_rel("C", "R1")
        self.assertListEqual(
            his1,
            [
                Fact("A", "R1", "C", "2024-01-02"),
                Fact("A", "R1", "C", "2024-01-06"),
            ]
        )
        tkg.construct_search_indices()
        his2 = tkg.find_history_by_tail_rel("C", "R1")
        self.assertListEqual(his1, his2)

    def test_anonymize_entity(self):
        """Check the anonymize entity method."""
        self.assertEqual(
            tkg.anonymize_entity("A"),
            "A"
        )
        tkg.anon_entity = "index"
        self.assertEqual(
            tkg.anonymize_entity("A"),
            "0",
        )
        tkg.anon_entity = "prefix"
        self.assertEqual(
            tkg.anonymize_entity("A"),
            "ENT_0"
        )

    def test_anonymize_relation(self):
        self.assertEqual(
            tkg.anonymize_rel("R1"),
            "R1"
        )
        tkg.anon_rel = "index"
        self.assertEqual(
            tkg.anonymize_rel("R1"),
            "0"
        )
        tkg.anon_rel = "prefix"
        self.assertEqual(
            tkg.anonymize_rel("R1"),
            "REL_0"
        )

    def test_anonymize_time(self):
        self.assertEqual(
            tkg.anonymize_time("2024-01-01"),
            "2024-01-01",
        )
        tkg.anon_time = "index"
        self.assertEqual(
            tkg.anonymize_time("2024-01-01"),
            "0"
        )
        tkg.anon_time = "suffix"
        self.assertEqual(
            tkg.anonymize_time("2024-01-02"),
            "on the 1st day"
        )

