import unittest
from datetime import datetime, timedelta

from src.prepare.prepare_tsro import construct_queries, construct_prompt
from src.utils.fact import Fact
from src.utils.query import Query
from src.utils.tkg import construct_search_histories, construct_time_index, TKG


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
time_unit = timedelta(days=1)
search_history = construct_search_histories(
    facts=train_facts + valid_facts + test_facts
)
time2id=construct_time_index(
    facts=train_facts + valid_facts + test_facts,
    base_time=base_time,
    time_unit=time_unit,
    time_precision="day",
)
tkg = TKG(
    name="test",
    train_facts=train_facts,
    valid_facts=valid_facts,
    test_facts=test_facts,
    base_time=base_time,
    time_unit=time_unit,
    time_precision="day",
    entities=entities,
    entity2id=entity2id,
    relations=relations,
    relation2id=relation2id,
    search_history=search_history,
    time2id=time2id,
)


class TestPrepareTsro(unittest.TestCase):
    def test_construct_queries(self):
        train_queries = construct_queries(train_facts)
        self.assertEqual(len(train_queries), 10)
        valid_queries = construct_queries(valid_facts)
        self.assertEqual(len(valid_queries), 8)
        test_queries = construct_queries(test_facts)
        self.assertEqual(len(test_queries), 5)

    def test_construct_prompt_entity_uni(self):
        test_queries = construct_queries(test_facts)
        # entity role is head
        query = test_queries[0]
        self.assertEqual(
            query,
            Query(
                entity="A",
                rel="R1",
                answers=["B", "C"],
                time="2024-01-06",
                entity_role="head",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
        )
        expected = "0:[A,R1,0.B]\n" \
                   "0:[A,R2,0.B]\n" \
                   "1:[A,R1,1.C]\n" \
                   "3:[A,R1,0.B]\n" \
                   "3:[A,R2,1.C]\n" \
                   "4:[A,R1,0.B]\n" \
                   "4:[A,R2,0.B]\n" \
                   "5:[A,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # entity role is tail
        query = test_queries[1]
        self.assertEqual(
            query,
            Query(
                entity="B",
                rel="R1",
                answers=["A"],
                time="2024-01-06",
                entity_role="tail",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
        )
        expected = "0:[B,R1,0.A]\n"\
                   "0:[B,R2,0.A]\n"\
                   "1:[B,R2,1.C]\n"\
                   "3:[B,R1,0.A]\n"\
                   "4:[B,R1,0.A]\n"\
                   "4:[B,R2,0.A]\n"\
                   "5:[B,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "C"})

    def test_construct_prompt_entity_bi(self):
        test_queries = construct_queries(test_facts)
        # entity role is tail
        query = test_queries[2]
        self.assertEqual(
            query,
            Query(
                entity="C",
                rel="R1",
                answers=["A"],
                time="2024-01-06",
                entity_role="tail",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="bi",
        )
        expected = "1:[C,R1,0.A]\n" \
                   "1:[C,R2,1.B]\n" \
                   "2:[C,R1,2.D]\n" \
                   "3:[C,R2,0.A]\n" \
                   "5:[C,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "B", "2": "D"})

    def test_construct_prompt_pair_uni(self):
        test_queries = construct_queries(test_facts)
        # entity role is head
        query = test_queries[0]
        self.assertEqual(
            query,
            Query(
                entity="A",
                rel="R1",
                answers=["B", "C"],
                time="2024-01-06",
                entity_role="head",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="pair",
            history_direction="uni",
        )
        expected = "0:[A,R1,0.B]\n" \
                   "1:[A,R1,1.C]\n" \
                   "3:[A,R1,0.B]\n" \
                   "4:[A,R1,0.B]\n" \
                   "5:[A,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # entity role is tail
        query = test_queries[1]
        self.assertEqual(
            query,
            Query(
                entity="B",
                rel="R1",
                answers=["A"],
                time="2024-01-06",
                entity_role="tail",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="pair",
            history_direction="uni",
        )
        expected = "0:[B,R1,0.A]\n" \
                   "3:[B,R1,0.A]\n" \
                   "4:[B,R1,0.A]\n" \
                   "5:[B,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A"})

    def test_construct_prompt_pair_bi(self):
        test_queries = construct_queries(test_facts)
        # entity role is tail
        query = test_queries[2]
        self.assertEqual(
            query,
            Query(
                entity="C",
                rel="R1",
                answers=["A"],
                time="2024-01-06",
                entity_role="tail",
            )
        )
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="pair",
            history_direction="bi",
        )
        expected = "1:[C,R1,0.A]\n" \
                   "2:[C,R1,1.D]\n" \
                   "5:[C,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "A", "1": "D"})

    def test_construct_prompt_anonymization(self):
        test_queries = construct_queries(test_facts)
        query = test_queries[0]
        self.assertEqual(
            query,
            Query(
                entity="A",
                rel="R1",
                answers=["B", "C"],
                time="2024-01-06",
                entity_role="head",
            )
        )
        # No anonymization
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
            anonymize_entity=False,
            anonymize_rel=False,
            anonymize_time=False,
        )
        expected = "2024-01-01:[A,R1,0.B]\n" \
                   "2024-01-01:[A,R2,0.B]\n" \
                   "2024-01-02:[A,R1,1.C]\n" \
                   "2024-01-04:[A,R1,0.B]\n" \
                   "2024-01-04:[A,R2,1.C]\n" \
                   "2024-01-05:[A,R1,0.B]\n" \
                   "2024-01-05:[A,R2,0.B]\n" \
                   "2024-01-06:[A,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Anonymize time
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
            anonymize_entity=False,
            anonymize_rel=False,
            anonymize_time=True,
        )
        expected = "0:[A,R1,0.B]\n" \
                   "0:[A,R2,0.B]\n" \
                   "1:[A,R1,1.C]\n" \
                   "3:[A,R1,0.B]\n" \
                   "3:[A,R2,1.C]\n" \
                   "4:[A,R1,0.B]\n" \
                   "4:[A,R2,0.B]\n" \
                   "5:[A,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "B", "1": "C"})
        # Anonymize entity
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
            anonymize_entity=True,
            anonymize_rel=False,
            anonymize_time=True,
        )
        expected = "0:[0,R1,0.1]\n" \
                   "0:[0,R2,0.1]\n" \
                   "1:[0,R1,1.2]\n" \
                   "3:[0,R1,0.1]\n" \
                   "3:[0,R2,1.2]\n" \
                   "4:[0,R1,0.1]\n" \
                   "4:[0,R2,0.1]\n" \
                   "5:[0,R1,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "1", "1": "2"})
        # Anonymize rel
        inputs = construct_prompt(
            query=query,
            tkg=tkg,
            history_length=30,
            history_type="entity",
            history_direction="uni",
            anonymize_entity=True,
            anonymize_rel=True,
            anonymize_time=True,
        )
        expected = "0:[0,0,0.1]\n" \
                   "0:[0,1,0.1]\n" \
                   "1:[0,0,1.2]\n" \
                   "3:[0,0,0.1]\n" \
                   "3:[0,1,1.2]\n" \
                   "4:[0,0,0.1]\n" \
                   "4:[0,1,0.1]\n" \
                   "5:[0,0,"
        self.assertEqual(inputs["inputs"], expected)
        self.assertDictEqual(inputs["candidates"], {"0": "1", "1": "2"})
