import unittest

from src.utils.metric import compute_rank, compute_hits

tot_preds = [
    ["B", "D", "E", "C"],
    ["C", "A"],
    ["B"]
]
tot_labels = [
    ["B", "C"],
    ["A"],
    ["A"]
]


class TestMetric(unittest.TestCase):
    def test_raw_rank(self):
        preds = ["B", "D", "E", "C"]
        labels = ["B", "C"]
        raw_rank = compute_rank(preds, labels, False)
        self.assertListEqual(
            raw_rank,
            [0, 3]
        )

    def test_filter_rank(self):
        preds = ["B", "D", "E", "C"]
        labels = ["B", "C"]
        raw_rank = compute_rank(preds, labels, True)
        self.assertListEqual(
            raw_rank,
            [0, 2]
        )

    def test_metric(self):
        metrics = compute_hits(tot_preds, tot_labels)
        self.assertDictEqual(
            metrics,
            {
                "raw hit@1": 0.25,
                "raw hit@3": 0.5,
                "raw hit@10": 0.75,
                "filter hit@1": 0.25,
                "filter hit@3": 0.75,
                "filter hit@10": 0.75,
            }
        )
