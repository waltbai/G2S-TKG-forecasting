import unittest
from datetime import datetime

from src.tsro.prepare import construct_queries, PrepareArguments, get_data_name, construct_prompt
from src.utils.common import DAY
from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import construct_search_histories, TKG

entities = ["A", "B", "C", "D"]
entity2id = {"A": 0, "B": 1, "C": 2, "D": 3}
relations = ["R1", "R2"]
relation2id = {"R1": 0, "R2": 1}
train_facts = [
    Fact("A", "R1", "B", "2024-01-01"),
    Fact("A", "R2", "B", "2024-01-01"),
    Fact("A", "R1", "C", "2024-01-02"),
    Fact("C", "R2", "B", "2024-01-02"),
    Fact("C", "R1", "D", "2024-01-03"),
]
valid_facts = [
    Fact("A", "R1", "B", "2024-01-04"),
    Fact("A", "R2", "C", "2024-01-04"),
    Fact("A", "R1", "B", "2024-01-05"),
    Fact("A", "R2", "B", "2024-01-05"),
]
test_facts = [
    Fact("A", "R1", "B", "2024-01-06"),
    Fact("A", "R1", "C", "2024-01-06"),
    Fact("A", "R1", "D", "2024-01-07"),
]
base_time = datetime.fromisoformat("2024-01-01")
time_unit = DAY
search_history = construct_search_histories(
    facts=train_facts + valid_facts + test_facts
)
time2id = {
    "2024-01-01": 0,
    "2024-01-02": 1,
    "2024-01-03": 2,
    "2024-01-04": 3,
    "2024-01-05": 4,
    "2024-01-06": 5,
    "2024-01-07": 6,
}
id2time = {v: k for k, v in time2id.items()}
tkg = TKG(
    name="test",
    train_facts=train_facts,
    valid_facts=valid_facts,
    test_facts=test_facts,
    base_time=base_time,
    time_unit=time_unit,
    entities=entities,
    entity2id=entity2id,
    relations=relations,
    relation2id=relation2id,
    search_history=search_history,
    time2id=time2id,
    id2time=id2time,
)


class TestTsroPrepare(unittest.TestCase):
    """Test tsro format prepare functions."""

    def test_get_data_name(self):
        # Case 1: default
        args = PrepareArguments(
            dataset="ICEWS14",
            anonymize_entity=False,
            anonymize_rel=False,
            anonymize_time=True,
            history_length=30,
            history_type="entity",
            history_direction="uni",
        )
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-anonymize_time-30-entity-uni.json"
        )
        # Case 2: change anonymize_entity
        args.anonymize_entity = True
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-anonymize_entity-anonymize_time-30-entity-uni.json"
        )
        # Case 3: change anonymize_rel
        args.anonymize_rel = True
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-"
            "anonymize_entity-anonymize_rel-anonymize_time-"
            "30-entity-uni.json"
        )
        # Case 4: change anonymize_time
        args.anonymize_time = False
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-"
            "anonymize_entity-anonymize_rel-"
            "30-entity-uni.json"
        )
        # Case 5: change history_length
        args.history_length = 50
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-"
            "anonymize_entity-anonymize_rel-"
            "50-entity-uni.json"
        )
        # Case 6: change history_type
        args.history_type = "pair"
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-"
            "anonymize_entity-anonymize_rel-"
            "50-pair-uni.json"
        )
        # Case 7: change history_direction
        args.history_direction = "bi"
        self.assertEqual(
            get_data_name(args),
            "ICEWS14-tsro-train-valid-test-"
            "anonymize_entity-anonymize_rel-"
            "50-pair-bi.json"
        )
        # Case 8: change prepare_parts
        self.assertEqual(
            get_data_name(args, ["test"]),
            "ICEWS14-tsro-test-"
            "anonymize_entity-anonymize_rel-"
            "50-pair-bi.json"
        )
        self.assertEqual(
            get_data_name(args, ["train", "valid"]),
            "ICEWS14-tsro-train-valid-"
            "anonymize_entity-anonymize_rel-"
            "50-pair-bi.json"
        )

    def test_construct_queries(self):
        train_queries = construct_queries(train_facts)
        self.assertEqual(len(train_queries), 10)
        valid_queries = construct_queries(valid_facts)
        self.assertEqual(len(valid_queries), 8)
        test_queries = construct_queries(test_facts)
        self.assertEqual(len(test_queries), 6)

    def test_construct_prompt_entity_uni(self):
        test_queries = construct_queries(test_facts)
        params = {
            "anonymize_entity": False,
            "anonymize_rel": False,
            "anonymize_time": True,
            "history_length": 30,
            "history_type": "entity",
            "history_direction": "uni",
        }
        # Case 1: entity_role == "head"
        query = test_queries[0]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[A,R1,0.B]\n" \
            "0:[A,R2,0.B]\n" \
            "1:[A,R1,1.C]\n" \
            "3:[A,R1,0.B]\n" \
            "3:[A,R2,1.C]\n" \
            "4:[A,R1,0.B]\n" \
            "4:[A,R2,0.B]\n" \
            "5:[A,R1,"
        self.assertEqual(query, Query("A", "R1", "B", "2024-01-06", "head", ["C"]))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Case 2: entity_role == "tail"
        query = test_queries[2]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[B,R1,0.A]\n" \
            "0:[B,R2,0.A]\n" \
            "1:[B,R2,1.C]\n" \
            "3:[B,R1,0.A]\n" \
            "4:[B,R1,0.A]\n" \
            "4:[B,R2,0.A]\n" \
            "5:[B,R1,"
        self.assertEqual(query, Query("B", "R1", "A", "2024-01-06", "tail", []))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "C"})

    def test_construct_prompt_entity_bi(self):
        test_queries = construct_queries(test_facts)
        params = {
            "anonymize_entity": False,
            "anonymize_rel": False,
            "anonymize_time": True,
            "history_length": 30,
            "history_type": "entity",
            "history_direction": "bi",
        }
        # Case 1: entity_role == "tail"
        query = test_queries[3]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "1:[C,R1,0.A]\n" \
            "1:[C,R2,1.B]\n" \
            "2:[C,R1,2.D]\n" \
            "3:[C,R2,0.A]\n" \
            "5:[C,R1,"
        self.assertEqual(query, Query("C", "R1", "A", "2024-01-06", "tail", []))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "B", "2": "D"})

    def test_construct_prompt_pair_uni(self):
        test_queries = construct_queries(test_facts)
        params = {
            "anonymize_entity": False,
            "anonymize_rel": False,
            "anonymize_time": True,
            "history_length": 30,
            "history_type": "pair",
            "history_direction": "uni",
        }
        # Case 1: entity_role == "head"
        query = test_queries[0]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[A,R1,0.B]\n" \
            "1:[A,R1,1.C]\n" \
            "3:[A,R1,0.B]\n" \
            "4:[A,R1,0.B]\n" \
            "5:[A,R1,"
        self.assertEqual(query, Query("A", "R1", "B", "2024-01-06", "head", ["C"]))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Case 2: entity_role == "tail"
        query = test_queries[2]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[B,R1,0.A]\n" \
            "3:[B,R1,0.A]\n" \
            "4:[B,R1,0.A]\n" \
            "5:[B,R1,"
        self.assertEqual(query, Query("B", "R1", "A", "2024-01-06", "tail", []))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A"})

    def test_construct_prompt_pair_bi(self):
        test_queries = construct_queries(test_facts)
        params = {
            "anonymize_entity": False,
            "anonymize_rel": False,
            "anonymize_time": True,
            "history_length": 30,
            "history_type": "pair",
            "history_direction": "bi",
        }
        # Case 1: entity_role == "head"
        query = test_queries[3]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "1:[C,R1,0.A]\n" \
            "2:[C,R1,1.D]\n" \
            "5:[C,R1,"
        self.assertEqual(query, Query("C", "R1", "A", "2024-01-06", "tail", []))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "D"})

    def test_construct_prompt_anonymize(self):
        test_queries = construct_queries(test_facts)
        params = {
            "anonymize_entity": False,
            "anonymize_rel": False,
            "anonymize_time": False,
            "history_length": 30,
            "history_type": "entity",
            "history_direction": "uni",
        }
        # Case 1: no anonymize
        query = test_queries[0]
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "2024-01-01:[A,R1,0.B]\n" \
            "2024-01-01:[A,R2,0.B]\n" \
            "2024-01-02:[A,R1,1.C]\n" \
            "2024-01-04:[A,R1,0.B]\n" \
            "2024-01-04:[A,R2,1.C]\n" \
            "2024-01-05:[A,R1,0.B]\n" \
            "2024-01-05:[A,R2,0.B]\n" \
            "2024-01-06:[A,R1,"
        self.assertEqual(query, Query("A", "R1", "B", "2024-01-06", "head", ["C"]))
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Case 2: anonymize time
        params["anonymize_time"] = True
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[A,R1,0.B]\n" \
            "0:[A,R2,0.B]\n" \
            "1:[A,R1,1.C]\n" \
            "3:[A,R1,0.B]\n" \
            "3:[A,R2,1.C]\n" \
            "4:[A,R1,0.B]\n" \
            "4:[A,R2,0.B]\n" \
            "5:[A,R1,"
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Case 3: anonymize entity
        params["anonymize_entity"] = True
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[0,R1,0.1]\n" \
            "0:[0,R2,0.1]\n" \
            "1:[0,R1,1.2]\n" \
            "3:[0,R1,0.1]\n" \
            "3:[0,R2,1.2]\n" \
            "4:[0,R1,0.1]\n" \
            "4:[0,R2,0.1]\n" \
            "5:[0,R1,"
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "1", "1": "2"})
        # Case 4: anonymize relation
        params["anonymize_rel"] = True
        inputs = construct_prompt(query, tkg, **params)
        expected = \
            "0:[0,0,0.1]\n" \
            "0:[0,1,0.1]\n" \
            "1:[0,0,1.2]\n" \
            "3:[0,0,0.1]\n" \
            "3:[0,1,1.2]\n" \
            "4:[0,0,0.1]\n" \
            "4:[0,1,0.1]\n" \
            "5:[0,0,"
        self.assertEqual(inputs["prompt"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "1", "1": "2"})
