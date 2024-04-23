import unittest

from src.evaluation import metric, Query


queries = [
    Query(
        entity="A",
        rel="R1",
        answers=["B", "C"],
        time="1",
        direction="tail",
        predictions=["B", "D", "E", "C"],
    ),
    Query(
        entity="B",
        rel="R1",
        answers=["A"],
        time="1",
        direction="head",
        predictions=["C", "A"],
    ),
    Query(
        entity="C",
        rel="R1",
        answers=["A"],
        time="1",
        direction="head",
        predictions=["A", "B"],
    )
]


class TestMetric(unittest.TestCase):
    """Test evaluation metric functions."""

    def test_metric_raw(self):
        metrics = metric(queries, time_filter=False)
        self.assertDictEqual(
            metrics,
            {
                "hit@1": 0.5,
                "hit@3": 0.75,
                "hit@10": 1.0
            }
        )

    def test_metric_time_filter(self):
        metrics = metric(queries, time_filter=True)
        self.assertDictEqual(
            metrics,
            {
                "hit@1": 0.5,
                "hit@3": 1.0,
                "hit@10": 1.0,
            }
        )
