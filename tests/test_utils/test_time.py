import unittest
from datetime import datetime

from src.utils.common import (
    MINUTE_15,
    DAY,
    YEAR,
    time_id2str,
    time_str2id,
)


class TestTime(unittest.TestCase):
    """Test time functions."""

    def test_time_id2str(self):
        base_time = datetime.fromisoformat("2000-01-01")
        time_unit = DAY
        # Case 1: self
        self.assertEqual(
            time_id2str(0, base_time, time_unit),
            "2000-01-01"
        )
        # Case 2: idx=1
        self.assertEqual(
            time_id2str(1, base_time, time_unit),
            "2000-01-02"
        )
        # Case 3: change time unit to year
        norm_factor = 1
        time_unit = YEAR
        self.assertEqual(
            time_id2str(1, base_time, time_unit),
            "2001"
        )
        # Case 4: time_unit == MINUTE_15, idx = 1
        time_unit = MINUTE_15
        self.assertEqual(
            time_id2str(1, base_time, time_unit),
            "2000-01-01 00:15:00"
        )

    def test_time_str2id(self):
        base_time = datetime.fromisoformat("2000-01-01")
        time_unit = DAY
        # Case 1: self
        self.assertEqual(time_str2id("2000-01-01", base_time, time_unit), 0)
        # Case 2: idx = 1
        self.assertEqual(time_str2id("2000-01-02", base_time, time_unit), 1)
        # Case 3: time_unit == YEAR
        time_unit = YEAR
        self.assertEqual(time_str2id("2001", base_time, time_unit), 1)
        # Case 4: time_unit == MINUTE_15
        time_unit = MINUTE_15
        self.assertEqual(
            time_str2id("2000-01-01 00:30:00", base_time, time_unit),
            2
        )
