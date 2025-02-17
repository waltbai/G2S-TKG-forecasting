import unittest

from src.query import Query
from src.tkg import TKG
from src.prompt import PromptConstructor


class TestPromptConstructor(unittest.TestCase):
    """
    Unit test for PromptConstructor class.
    """

    def test_prompt(self):
        tkg = TKG.load("tests", "tkg")
        prompt_constructor = PromptConstructor()

        # Check head query
        query = Query(
            entity="A",
            rel="R1",
            answer="B",
            time=5,
            role="head"
        )
        tkg.find_history(query, "hop")
        prompt_constructor(query, tkg)
        self.assertEqual(
            query.prompt,
            ("### Entities ###\n"
             "0:A\n"
             "1:B\n"
             "2:C\n\n"
             "### Relations ###\n"
             "0:R1\n"
             "1:R2\n\n"
             "### History ###\n"
             "0:[0,0,1]\n"
             "0:[0,1,1]\n"
             "1:[0,0,2]\n"
             "3:[0,0,1]\n"
             "3:[0,1,2]\n"
             "4:[0,0,1]\n"
             "4:[0,1,1]\n\n"
             "### Query ###\n"
             "5:[0,0,?]\n\n"
             "### Answer ###\n")
        )
        self.assertEqual(
            query.label,
            "1"
        )

        # Check tail query
        query = Query(
            entity="B",
            rel="R1",
            answer="A",
            time=5,
            role="tail"
        )
        tkg.find_history(query, "hop")
        prompt_constructor(query, tkg)
        self.assertEqual(
            query.prompt,
            ("### Entities ###\n"
             "0:A\n"
             "1:B\n"
             "2:C\n\n"
             "### Relations ###\n"
             "0:R1\n"
             "1:R2\n\n"
             "### History ###\n"
             "0:[0,0,1]\n"
             "0:[0,1,1]\n"
             "1:[2,1,1]\n"
             "3:[0,0,1]\n"
             "4:[0,0,1]\n"
             "4:[0,1,1]\n\n"
             "### Query ###\n"
             "5:[?,0,1]\n\n"
             "### Answer ###\n")
        )
        self.assertEqual(
            query.label,
            "0"
        )

