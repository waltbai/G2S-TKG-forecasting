import unittest

from src.utils.metric import compute_rank, compute_hits

tot_preds = [
    ["B", "D", "E", "C"],
    ["B", "D", "E", "C"],
    ["C", "A"],
    ["B"]
]
tot_answers = ["B", "C", "A", "A"]
tot_filters = [["C"], ["B"], [], []]


class TestMetric(unittest.TestCase):
    """Test metrics."""

    def test_compute_rank_raw(self):
        rank = compute_rank(tot_preds[0], tot_answers[0])
        self.assertEqual(rank, 0)
        rank = compute_rank(tot_preds[1], tot_answers[1])
        self.assertEqual(rank, 3)

    def test_compute_rank_filter(self):
        rank = compute_rank(tot_preds[0], tot_answers[0], tot_filters[0])
        self.assertEqual(rank, 0)
        rank = compute_rank(tot_preds[1], tot_answers[1], tot_filters[1])
        self.assertEqual(rank, 2)

    def test_compute_hits(self):
        metrics = compute_hits(tot_preds, tot_answers, tot_filters)
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
