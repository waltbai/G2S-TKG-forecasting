import unittest

from src.args import AnonymizedDataArguments
from src.stage1.prepare import get_data_version


class TestGetVersion(unittest.TestCase):
    def test_get_version(self):
        # Case 1: single train, valid, test
        args = AnonymizedDataArguments(
            train_dataset="ICEWS14",
            valid_dataset="ICEWS14",
            test_dataset="ICEWS14",
        )
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14-valid_ICEWS14-test_ICEWS14"
            "-entity-uni-30-global-start-inline"
        )

        # Case 2: multi train, single valid, test
        args.train_dataset = "ICEWS14,ICEWS18"
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14_ICEWS18-valid_ICEWS14-test_ICEWS14"
            "-entity-uni-30-global-start-inline"
        )

        # Case 3: multi train, valid, test
        args.valid_dataset = "ICEWS14,ICEWS18"
        args.test_dataset = "ICEWS14,ICEWS18"
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14_ICEWS18"
            "-valid_ICEWS14_ICEWS18-test_ICEWS14_ICEWS18"
            "-entity-uni-30-global-start-inline"
        )

        # Case 4: with prefix
        args.anonymize_prefix = True
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14_ICEWS18"
            "-valid_ICEWS14_ICEWS18-test_ICEWS14_ICEWS18"
            "-entity-uni-30-global_prefix-start-inline"
        )

        # Case 4: vague time
        args.time_process_strategy = "query"
        args.vague_time = True
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14_ICEWS18"
            "-valid_ICEWS14_ICEWS18-test_ICEWS14_ICEWS18"
            "-entity-uni-30-global_prefix-query_vague-inline"
        )

        # Case 5: qa template
        args.prompt_construct_strategy = "qa"
        self.assertEqual(
            get_data_version(args),
            "stage1-train_ICEWS14_ICEWS18"
            "-valid_ICEWS14_ICEWS18-test_ICEWS14_ICEWS18"
            "-entity-uni-30-global_prefix-query_vague-qa"
        )
