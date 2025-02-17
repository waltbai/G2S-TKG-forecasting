import unittest

from src.metric import compute_metrics


class TestMetric(unittest.TestCase):
    """
    Test metric functions
    """

    def test_metric(self):
        tot_preds = [
            ["B", "D", "E", "C"],
            ["B", "D", "E", "C"],
            ["C", "A"],
            ["B"]
        ]
        tot_answers = ["B", "C", "A", "A"]
        tot_filters = [["C"], ["B"], [], []]
        # Test hit without filter
        metrics = compute_metrics(
            tot_preds=tot_preds,
            tot_answers=tot_answers,
            tot_filters=tot_filters,
        )
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
