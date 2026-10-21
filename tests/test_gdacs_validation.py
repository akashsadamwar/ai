"""Standalone regressions for the validation fixes in the pasted adapter."""
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from services import gdacs_floods as gdacs


class FloodValidationTests(unittest.TestCase):
    def fetch(self, ring=None, **properties):
        feature = {
            "properties": {
                "eventid": 1, "episodeid": 2, "name": "Test flood",
                "episodealertlevel": "Orange", "iscurrent": "false",
                "fromdate": "2026-10-01T00:00:00",
                "todate": "2026-10-20T00:00:00", **properties,
            },
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        }
        if ring is None:
            ring = [[0, 0], [1, 0], [0, 1], [0, 0]]
        area = {"features": [{
            "properties": {"eventid": 1, "episodeid": 2, "Class": "Poly_Affected"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }]}
        with patch.object(gdacs, "_get", side_effect=[{"features": [feature]}, area]):
            return gdacs.fetch_events(0, 0, 1050, datetime(2026, 10, 10, tzinfo=timezone.utc))

    def test_reversed_dates_fail_instead_of_silently_skipping(self):
        with self.assertRaisesRegex(gdacs.FeedError, "end date precedes"):
            self.fetch(fromdate="2026-10-15T00:00:00", todate="2026-10-05T00:00:00")

    def test_unclosed_ring_fails(self):
        with self.assertRaisesRegex(gdacs.FeedError, "not closed"):
            self.fetch(ring=[[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_repeated_points_fail(self):
        with self.assertRaisesRegex(gdacs.FeedError, "distinct positions"):
            self.fetch(ring=[[0, 0], [1, 0], [1, 0], [0, 0]])

    def test_valid_ring_and_dates_are_preserved(self):
        event, = self.fetch()
        self.assertEqual(event["area"]["coordinates"], [[[[0, 0], [1, 0], [0, 1], [0, 0]]]])
        self.assertEqual(event["expires_at"], datetime(2026, 10, 20, tzinfo=timezone.utc))

    def test_running_flood_with_stale_update_remains_open(self):
        event, = self.fetch(iscurrent="true", todate="2026-10-05T00:00:00")
        self.assertIsNone(event["expires_at"])

    def test_altitude_is_removed_before_checking_closure(self):
        event, = self.fetch(ring=[[0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 4]])
        self.assertEqual(event["area"]["coordinates"][0][0][0], [0, 0])


if __name__ == "__main__":
    unittest.main()
