import unittest
from unittest import mock

from lab.mt01 import event_order, fixture_snapshots, replay, replay_session, summarize


class OfflineSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replayed = replay()
        cls.sessions = cls.replayed["sessions"]

    def rows(self, name):
        return self.sessions[name]["turns"]

    def test_every_session_matches_its_expected_result(self):
        for name, session in self.sessions.items():
            with self.subTest(session=name):
                self.assertEqual(session["summary"], session["expected"])
        self.assertTrue(self.replayed["all_expected"])

    def test_regression(self):
        rows = self.rows("regression")
        self.assertEqual([r["marker"] for r in rows], ["exact", "absent", "exact", "exact"])
        self.assertEqual([r["invariant"] for r in rows], ["pass", "pass", "pass", "fail"])

    def test_clean_control_differs_only_in_turn_4_snapshot(self):
        reg, ctl = self.rows("regression"), self.rows("clean_control")
        self.assertEqual([r["marker"] for r in reg], [r["marker"] for r in ctl])
        self.assertEqual([r["snapshot"] for r in reg[:3]], [r["snapshot"] for r in ctl[:3]])
        self.assertNotEqual(reg[3]["snapshot"], ctl[3]["snapshot"])
        self.assertEqual([r["invariant"] for r in ctl], ["pass"] * 4)
        self.assertEqual(ctl[3]["checks"]["batch.feature"], "pass")

    def test_unobserved_turn_keeps_snapshot_and_leaves_order_unknown(self):
        rows = self.rows("unobserved_response")
        self.assertEqual([r["marker"] for r in rows], ["exact", "exact", "unobserved", "exact"])
        self.assertEqual(rows[2]["invariant"], "pass")
        self.assertEqual(self.sessions["unobserved_response"]["summary"]["event_order"], "unknown")

    def test_features_apply_from_their_turn(self):
        for name in self.sessions:
            for row in self.rows(name):
                for task, start in (("reschedule", 1), ("availability", 2), ("batch", 4)):
                    with self.subTest(session=name, turn=row["turn"], task=task):
                        self.assertEqual(task + ".feature" in row["checks"], row["turn"] >= start)
                        if row["turn"] >= start:
                            self.assertEqual(row["checks"][task + ".feature"], "pass")
        snapshots = fixture_snapshots()
        self.assertFalse(hasattr(snapshots["reschedule"], "available_rooms"))
        self.assertFalse(hasattr(snapshots["availability"], "book_many"))


class Summary(unittest.TestCase):
    def test_malformed_counts_as_noncompliance(self):
        rows = [{"turn": 1, "invariant": "pass", "marker": "exact"},
                {"turn": 2, "invariant": "pass", "marker": "malformed"},
                {"turn": 3, "invariant": "fail", "marker": "absent"}]
        summary = summarize(rows)
        self.assertEqual(summary["first_observed_noncompliance"], 2)
        self.assertEqual(summary["first_noncompliance_kind"], "malformed")
        self.assertEqual(summary["event_order"], "noncompliance_first")

    def test_initial_failure_is_not_a_regression(self):
        rows = [{"turn": 1, "invariant": "fail", "marker": "exact"},
                {"turn": 2, "invariant": "pass", "marker": "exact"}]
        self.assertIsNone(summarize(rows)["first_observed_regression"])
        rows.append({"turn": 3, "invariant": "fail", "marker": "unobserved"})
        summary = summarize(rows)
        self.assertEqual(summary["first_observed_regression"], 3)
        self.assertEqual(summary["marker_at_regression"], "unobserved")
        self.assertEqual(summary["event_order"], "unknown")

    def test_only_unobserved_turns_create_uncertainty(self):
        rows = [{"turn": 1, "invariant": "pass", "marker": "exact"},
                {"turn": 2, "invariant": "pass", "marker": "exact"},
                {"turn": 3, "invariant": "fail", "marker": "exact"}]
        self.assertEqual(summarize(rows)["event_order"], "regression_first")
        rows[1]["marker"] = "unobserved"
        self.assertEqual(summarize(rows)["event_order"], "unknown")

    def test_evaluator_error_stops_replay(self):
        fixture = {"token": "0123456789abcdef", "invariant": "reschedule.conflict_rollback",
                   "applies_from": {"reschedule": 1}}
        session = {"turns": [{"snapshot": "any", "response": None}], "expected": {}}
        failed = {"status": "evaluator_error", "errors": [{"message": "boom"}], "checks": []}
        with mock.patch("lab.mt01.evaluate_factory", return_value=failed):
            with self.assertRaises(RuntimeError):
                replay_session(fixture, {"any": object}, session)

    def test_event_order_cases(self):
        cases = [
            ((2, 4, []), "noncompliance_first"),
            ((2, 4, [1]), "noncompliance_first"),  # an earlier hidden omission is still first
            ((4, 4, []), "same_turn"),
            ((4, 4, [3]), "unknown"),
            ((4, 4, [4]), "unknown"),  # inconsistent input still never claims same_turn
            ((5, 3, []), "regression_first"),
            ((None, 3, []), "regression_first"),
            ((None, 3, [3]), "unknown"),
            ((None, 3, [2]), "unknown"),
            ((None, 3, [4]), "regression_first"),  # a later gap cannot precede the regression
            ((2, None, []), "noncompliance_only"),
            ((None, None, []), "neither"),
            ((None, None, [2]), "neither_observed"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(event_order(*args), expected)


if __name__ == "__main__":
    unittest.main()
