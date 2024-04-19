import unittest

from src.evaluation import metric, Query


class TestQuery(unittest.TestCase):
    """Test query class."""

    def test_quadruple_prompt(self):
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-01",
            entity_idx=0,
            rel_idx=0,
            answers_idx=[1, 2],
            time_idx=0,
            direction="tail",
        )
        self.assertTupleEqual(
            query.prompt_quadruple(anonymize=False, anonymize_time=False),
            ("A", "R1", "2024-01-01")
        )
        self.assertTupleEqual(
            query.prompt_quadruple(anonymize=False, anonymize_time=True),
            ("A", "R1", "0")
        )
        self.assertTupleEqual(
            query.prompt_quadruple(anonymize=True, anonymize_time=False),
            ("0", "0", "2024-01-01")
        )
        self.assertTupleEqual(
            query.prompt_quadruple(anonymize=True, anonymize_time=True),
            ("0", "0", "0")
        )

    def test_text_prompt(self):
        query = Query(
            entity="A",
            rel="R1",
            answers=["B", "C"],
            time="2024-01-01",
            entity_idx=0,
            rel_idx=0,
            answers_idx=[1, 2],
            time_idx=0,
            direction="tail",
        )
        self.assertTupleEqual(
            query.prompt_text(anonymize=False, anonymize_time=False),
            ("A", "R1", "2024-01-01")
        )
        self.assertTupleEqual(
            query.prompt_text(anonymize=False, anonymize_time=True),
            ("A", "R1", "0th")
        )
        self.assertTupleEqual(
            query.prompt_text(anonymize=True, anonymize_time=False),
            ("0", "0", "2024-01-01")
        )
        self.assertTupleEqual(
            query.prompt_text(anonymize=True, anonymize_time=True),
            ("0", "0", "0th")
        )


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
