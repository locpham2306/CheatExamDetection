import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import config
from behavior_engine import BehaviorEngine


def make_person(track_id, box, head_direction="CENTER"):
    x1, y1, x2, y2 = box
    return {
        "track_id": track_id,
        "box": box,
        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
        "conf": 0.9,
        "head_direction": head_direction,
    }


def make_phone(center):
    x, y = center
    return {
        "box": [x - 5, y - 10, x + 5, y + 10],
        "center": center,
        "conf": 0.9,
    }


class BehaviorEngineTests(unittest.TestCase):
    def test_phone_is_assigned_only_to_closest_person(self):
        engine = BehaviorEngine()
        persons = [
            make_person(1, [0, 0, 100, 200]),
            make_person(2, [90, 0, 190, 200]),
        ]

        results = engine.process_frame_data(persons, [make_phone((105, 100))])
        by_id = {result["track_id"]: result for result in results}

        self.assertFalse(by_id[1]["has_phone"])
        self.assertTrue(by_id[2]["has_phone"])

    def test_sustained_phone_and_head_turn_reaches_high_risk(self):
        engine = BehaviorEngine()
        person = make_person(1, [0, 0, 100, 200], "LEFT")
        result = None

        for _ in range(config.TEMPORAL_WINDOW_FRAMES):
            result = engine.process_frame_data([person], [make_phone((50, 100))])[0]

        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["risk_score"], 100)

    def test_stale_track_history_is_removed(self):
        engine = BehaviorEngine()
        person = make_person(7, [0, 0, 100, 200])
        engine.process_frame_data([person], [])

        for _ in range(config.TRACK_HISTORY_TTL_FRAMES + 1):
            engine.process_frame_data([], [])

        self.assertNotIn(7, engine.history)
        self.assertNotIn(7, engine.last_seen)


if __name__ == "__main__":
    unittest.main()
