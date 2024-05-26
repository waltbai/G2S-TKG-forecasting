import unittest
from datetime import datetime

from src.stage1.anonymizer import get_anonymizer
from src.stage1.prepare import construct_queries, construct_history
from src.stage1.prompt import get_prompt_constructor
from src.stage1.time_processor import get_time_processor
from src.utils.fact import Fact
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
facts = train_facts + valid_facts + test_facts
search_history = construct_search_histories(facts)
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
    time_unit="day",
    entities=entities,
    entity2id=entity2id,
    relations=relations,
    relation2id=relation2id,
    search_history=search_history,
    time2id=time2id,
    id2time=id2time,
)


class TestInlinePromptConstructor(unittest.TestCase):
    def test_global_absolute(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("absolute")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix, sep is comma
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "2024-01-01:[0,0,0.1]\n" \
            "2024-01-01:[0,1,0.1]\n" \
            "2024-01-02:[0,0,1.2]\n" \
            "2024-01-04:[0,0,0.1]\n" \
            "2024-01-04:[0,1,1.2]\n" \
            "2024-01-05:[0,0,0.1]\n" \
            "2024-01-05:[0,1,0.1]\n" \
            "2024-01-06:[0,0,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")

        # Case 2: with prefix, sep is comma
        anonymizer(queries, tkg, True)
        prompt_constructor(queries)
        expected_prompt = \
            "2024-01-01:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-01:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-02:[ENT_0,REL_0,1.ENT_2]\n" \
            "2024-01-04:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-04:[ENT_0,REL_1,1.ENT_2]\n" \
            "2024-01-05:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-06:[ENT_0,REL_0,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "ENT_0", "B": "ENT_1", "C": "ENT_2"}
        )

    def test_global_start(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("start")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "0:[0,0,0.1]\n" \
            "0:[0,1,0.1]\n" \
            "1:[0,0,1.2]\n" \
            "3:[0,0,0.1]\n" \
            "3:[0,1,1.2]\n" \
            "4:[0,0,0.1]\n" \
            "4:[0,1,0.1]\n" \
            "5:[0,0,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")

    def test_global_query(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("query")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "5:[0,0,0.1]\n" \
            "5:[0,1,0.1]\n" \
            "4:[0,0,1.2]\n" \
            "2:[0,0,0.1]\n" \
            "2:[0,1,1.2]\n" \
            "1:[0,0,0.1]\n" \
            "1:[0,1,0.1]\n" \
            "0:[0,0,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")

    def test_session_absolute(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("absolute")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "2024-01-01:[0,0,0.1]\n" \
            "2024-01-01:[0,1,0.1]\n" \
            "2024-01-02:[0,1,1.2]\n" \
            "2024-01-04:[0,0,0.1]\n" \
            "2024-01-05:[0,0,0.1]\n" \
            "2024-01-05:[0,1,0.1]\n" \
            "2024-01-06:[0,0,"
        self.assertEqual(queries[2].prompt, expected_prompt)
        self.assertEqual(queries[2].label, "0")

        # Case 2: with prefix
        anonymizer(queries, tkg, True)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "2024-01-01:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-01:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-02:[ENT_0,REL_1,1.ENT_2]\n" \
            "2024-01-04:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-06:[ENT_0,REL_0,"
        self.assertEqual(queries[2].prompt, expected_prompt)
        self.assertEqual(queries[2].label, "0")
        self.assertDictEqual(
            queries[2].entity_mapping,
            {"B": "ENT_0", "A": "ENT_1", "C": "ENT_2"}
        )

    def test_session_start(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("start")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "0:[0,0,0.1]\n" \
            "0:[0,1,0.1]\n" \
            "1:[0,1,1.2]\n" \
            "3:[0,0,0.1]\n" \
            "4:[0,0,0.1]\n" \
            "4:[0,1,0.1]\n" \
            "5:[0,0,"
        self.assertEqual(queries[2].prompt, expected_prompt)
        self.assertEqual(queries[2].label, "0")

    def test_session_query(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("query")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "5:[0,0,0.1]\n" \
            "5:[0,1,0.1]\n" \
            "4:[0,1,1.2]\n" \
            "2:[0,0,0.1]\n" \
            "1:[0,0,0.1]\n" \
            "1:[0,1,0.1]\n" \
            "0:[0,0,"
        self.assertEqual(queries[2].prompt, expected_prompt)
        self.assertEqual(queries[2].label, "0")

    def test_origin_absolute(self):
        anonymizer = get_anonymizer("origin")
        time_processor = get_time_processor("absolute")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix, sep is comma
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "2024-01-01:[A,R1,0.B]\n" \
            "2024-01-01:[A,R2,0.B]\n" \
            "2024-01-02:[A,R1,1.C]\n" \
            "2024-01-04:[A,R1,0.B]\n" \
            "2024-01-04:[A,R2,1.C]\n" \
            "2024-01-05:[A,R1,0.B]\n" \
            "2024-01-05:[A,R2,0.B]\n" \
            "2024-01-06:[A,R1,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")

    def test_origin_start(self):
        anonymizer = get_anonymizer("origin")
        time_processor = get_time_processor("start")
        prompt_constructor = get_prompt_constructor("inline")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix, sep is comma
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "0:[A,R1,0.B]\n" \
            "0:[A,R2,0.B]\n" \
            "1:[A,R1,1.C]\n" \
            "3:[A,R1,0.B]\n" \
            "3:[A,R2,1.C]\n" \
            "4:[A,R1,0.B]\n" \
            "4:[A,R2,0.B]\n" \
            "5:[A,R1,"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertEqual(queries[0].label, "0")


class TestQAPromptConstructor(unittest.TestCase):
    def test_global_absolute(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("absolute")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix, sep is comma
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "2024-01-01:[0,0,0.1]\n" \
            "2024-01-01:[0,1,0.1]\n" \
            "2024-01-02:[0,0,1.2]\n" \
            "2024-01-04:[0,0,0.1]\n" \
            "2024-01-04:[0,1,1.2]\n" \
            "2024-01-05:[0,0,0.1]\n" \
            "2024-01-05:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "2024-01-06:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[0].prompt, expected_prompt)

        # Case 2: with prefix, sep is comma
        anonymizer(queries, tkg, True)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "2024-01-01:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-01:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-02:[ENT_0,REL_0,1.ENT_2]\n" \
            "2024-01-04:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-04:[ENT_0,REL_1,1.ENT_2]\n" \
            "2024-01-05:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_1,0.ENT_1]\n" \
            "\n### Query ###\n" \
            "2024-01-06:[ENT_0,REL_0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[0].prompt, expected_prompt)
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "ENT_0", "B": "ENT_1", "C": "ENT_2"}
        )

    def test_global_start(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("start")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "0:[0,0,0.1]\n" \
            "0:[0,1,0.1]\n" \
            "1:[0,0,1.2]\n" \
            "3:[0,0,0.1]\n" \
            "3:[0,1,1.2]\n" \
            "4:[0,0,0.1]\n" \
            "4:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "5:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[0].prompt, expected_prompt)

    def test_global_query(self):
        anonymizer = get_anonymizer("global")
        time_processor = get_time_processor("query")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "5:[0,0,0.1]\n" \
            "5:[0,1,0.1]\n" \
            "4:[0,0,1.2]\n" \
            "2:[0,0,0.1]\n" \
            "2:[0,1,1.2]\n" \
            "1:[0,0,0.1]\n" \
            "1:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "0:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[0].prompt, expected_prompt)

    def test_session_absolute(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("absolute")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "2024-01-01:[0,0,0.1]\n" \
            "2024-01-01:[0,1,0.1]\n" \
            "2024-01-02:[0,1,1.2]\n" \
            "2024-01-04:[0,0,0.1]\n" \
            "2024-01-05:[0,0,0.1]\n" \
            "2024-01-05:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "2024-01-06:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[2].prompt, expected_prompt)

        # Case 2: with prefix
        anonymizer(queries, tkg, True)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "2024-01-01:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-01:[ENT_0,REL_1,0.ENT_1]\n" \
            "2024-01-02:[ENT_0,REL_1,1.ENT_2]\n" \
            "2024-01-04:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_0,0.ENT_1]\n" \
            "2024-01-05:[ENT_0,REL_1,0.ENT_1]\n" \
            "\n### Query ###\n" \
            "2024-01-06:[ENT_0,REL_0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[2].prompt, expected_prompt)
        self.assertDictEqual(
            queries[2].entity_mapping,
            {"B": "ENT_0", "A": "ENT_1", "C": "ENT_2"}
        )

    def test_session_start(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("start")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "0:[0,0,0.1]\n" \
            "0:[0,1,0.1]\n" \
            "1:[0,1,1.2]\n" \
            "3:[0,0,0.1]\n" \
            "4:[0,0,0.1]\n" \
            "4:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "5:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[2].prompt, expected_prompt)

    def test_session_query(self):
        anonymizer = get_anonymizer("session")
        time_processor = get_time_processor("query")
        prompt_constructor = get_prompt_constructor("qa")

        queries = construct_queries(test_facts)
        construct_history(queries, tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg)
        time_processor(queries, tkg)
        prompt_constructor(queries)
        expected_prompt = \
            "### History ###\n" \
            "5:[0,0,0.1]\n" \
            "5:[0,1,0.1]\n" \
            "4:[0,1,1.2]\n" \
            "2:[0,0,0.1]\n" \
            "1:[0,0,0.1]\n" \
            "1:[0,1,0.1]\n" \
            "\n### Query ###\n" \
            "0:[0,0,?]\n" \
            "\n### Answer ###\n"
        self.assertEqual(queries[2].prompt, expected_prompt)
