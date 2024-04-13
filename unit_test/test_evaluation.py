import unittest

from llm4tkg.evaluation import Prediction, QueryResult, metric
from llm4tkg.preprocess.fact import Fact


class TestEvaluation(unittest.TestCase):
    """Test evaluation functions."""

    def test_query_result_filter(self):
        predictions = [
            Prediction("B", 0.9),
            Prediction("C", 0.6),
            Prediction("D", 0.3),
            Prediction("E", 0.1),
        ]
        result = QueryResult(
            query=Fact("A", "R1", "C", "0", 0, 0, 2, 0),
            query_target="tail",
            predictions=predictions,
            candidates={"0": "B", "1": "C", "2": "D", "3": "E"},
            prompt="",
            answer="C",
        )
        time_filter_set = {
            ("A", "R1", "B", "0"),
            ("A", "R2", "C", "0"),
            ("A", "R1", "D", "1")
        }
        result.time_filter(time_filter_set)
        self.assertEqual(
            result.predictions,
            [
                Prediction("C", 0.6),
                Prediction("D", 0.3),
                Prediction("E", 0.1),
            ]
        )

    def test_query_result_filter_head(self):
        predictions = [
            Prediction("B", 0.9),
            Prediction("A", 0.6),
            Prediction("D", 0.3),
            Prediction("E", 0.1),
        ]
        result = QueryResult(
            query=Fact("A", "R1", "C", "0", 0, 0, 2, 0),
            query_target="head",
            predictions=predictions,
            candidates={"0": "B", "1": "A", "2": "D", "3": "E"},
            prompt="",
            answer="A",
        )
        time_filter_set = {
            ("B", "R1", "C", "0"),
            ("A", "R2", "C", "0"),
            ("A", "R1", "D", "1")
        }
        result.time_filter(time_filter_set)
        self.assertEqual(
            result.predictions,
            [
                Prediction("A", 0.6),
                Prediction("D", 0.3),
                Prediction("E", 0.1),
            ]
        )

    def test_metrics(self):
        results = [
            QueryResult(
                query=Fact("A", "R1", "B", "0", 0, 0, 1, 0),
                query_target="tail",
                predictions=[
                    Prediction("C", 0.9),
                    Prediction("B", 0.6),
                    Prediction("D", 0.3),
                ],
                candidates={"0": "B", "1": "C", "2": "D", "3": "E"},
                prompt="",
                answer="B",
            ),
            QueryResult(
                query=Fact("A", "R2", "C", "0", 0, 1, 2, 0),
                query_target="head",
                predictions=[
                    Prediction("A", 0.9),
                    Prediction("B", 0.6),
                    Prediction("D", 0.3),
                ],
                candidates={"0": "B", "1": "A", "2": "D", "3": "E"},
                prompt="",
                answer="A",
            ),
        ]
        self.assertDictEqual(
            metric(results=results),
            {"hit@1": 0.5, "hit@3": 1.0, "hit@10": 1.0}
        )
