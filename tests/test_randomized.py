"""Randomized differential checks, the error oracle, and mutation scoring."""

from pathlib import Path
import types
import unittest
from unittest.mock import patch

from evaluator import randomized
from evaluator.evaluate import evaluate_factory
from evaluator.oracle import OracleStore
from lab import mutation


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    module = types.ModuleType("reference_under_test")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module.BookingStore


def rec(identifier="a", room="cedar", start="2026-01-02T10:00:00Z", end="2026-01-02T11:00:00Z"):
    return {"id": identifier, "room": room, "start": start, "end": end}


class OracleTests(unittest.TestCase):
    def test_permitted_errors_include_every_applicable_code(self):
        store = OracleStore()
        store.book(rec())
        self.assertEqual(store.errors_for("book", (rec(),)), {"DUPLICATE_ID", "CONFLICT"})
        self.assertEqual(store.errors_for("book", (rec("b", "oak", "bad", "2026-01-02T11:00:00Z"),)),
                         {"UNKNOWN_ROOM", "INVALID_INPUT"})
        self.assertEqual(store.errors_for("book", (rec("b", end="2026-01-02T12:00:00Z"),)), {"CONFLICT"})
        self.assertEqual(store.errors_for("book", (rec("b", "maple"),)), set())
        self.assertEqual(store.errors_for("book", ({"id": "b"},)), {"INVALID_INPUT"})

    def test_reschedule_and_batch_error_sets(self):
        store = OracleStore()
        store.book(rec())
        store.book(rec("b", start="2026-01-02T12:00:00Z", end="2026-01-02T13:00:00Z"))
        self.assertEqual(store.errors_for("reschedule", ("a", "2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z")), set())
        self.assertEqual(store.errors_for("reschedule", ("a", "2026-01-02T12:30:00Z", "2026-01-02T13:30:00Z")), {"CONFLICT"})
        self.assertEqual(store.errors_for("reschedule", ("zz", "bad", "2026-01-02T13:30:00Z")), {"BOOKING_NOT_FOUND", "INVALID_INPUT"})
        self.assertEqual(store.errors_for("reschedule", ("", "2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z")), {"INVALID_INPUT"})
        within = [rec("c", "maple"), rec("d", "maple", "2026-01-02T10:30:00Z", "2026-01-02T11:30:00Z")]
        self.assertEqual(store.errors_for("book_many", (within,)), {"CONFLICT"})
        self.assertEqual(store.errors_for("book_many", ([rec("c", "maple"), rec("c", "maple", "2026-01-02T14:00:00Z", "2026-01-02T15:00:00Z")],)), {"DUPLICATE_ID"})
        self.assertEqual(store.errors_for("book_many", ((rec("c", "maple"),),)), {"INVALID_INPUT"})
        self.assertEqual(store.errors_for("book_many", ([],)), set())

    def test_oracle_store_agrees_with_its_error_sets_on_generated_sequences(self):
        for task in ("baseline", "reschedule", "availability", "batch"):
            for seed in range(30):
                store = OracleStore()
                for operation, args in randomized.generate(task, seed):
                    allowed = store.errors_for(operation, args)
                    try:
                        getattr(store, operation)(*args)
                    except ValueError as error:
                        self.assertIn(str(error), allowed, (task, seed, operation, args))
                    else:
                        self.assertEqual(allowed, set(), (task, seed, operation, args))


class RandomizedCheckTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(randomized.generate("batch", 7), randomized.generate("batch", 7))
        self.assertNotEqual(randomized.generate("batch", 7), randomized.generate("batch", 8))

    def test_every_reference_passes(self):
        for task, path in mutation.SOURCES.items():
            with self.subTest(task=task):
                self.assertIsNone(randomized.check(load(path), task))

    def test_failure_reports_a_shrunk_reproduction(self):
        source = mutation.apply("reschedule", '"room": current["room"],', '"room": "cedar",')
        module = types.ModuleType("room_mutant")
        exec(compile(source, "<room mutant>", "exec"), module.__dict__)
        message = randomized.check(module.BookingStore, "reschedule")
        self.assertIsNotNone(message)
        self.assertIn("Minimal sequence", message)
        steps = int(message.split("Minimal sequence (")[1].split(" steps")[0])
        self.assertLessEqual(steps, 3)

    def test_oracle_defect_is_an_evaluator_error_not_a_submission_failure(self):
        with patch.object(OracleStore, "errors_for", return_value={"CONFLICT"}):
            result = evaluate_factory(load(mutation.SOURCES["baseline"]), "baseline")
        check = next(c for c in result["checks"] if c["id"] == "baseline.randomized_differential")
        self.assertEqual(check["status"], "unevaluable")
        self.assertEqual(result["status"], "evaluator_error")


class MutationTests(unittest.TestCase):
    def test_every_non_equivalent_mutant_is_killed_and_references_pass(self):
        summary = mutation.run()
        self.assertTrue(summary["references_pass"], summary["reference_failures"])
        self.assertEqual(summary["survivors"], [])
        self.assertEqual(summary["evaluator_errors"], [])
        self.assertEqual(summary["killed"], summary["scored"])
        self.assertEqual(summary["equivalent_but_killed"], [])
        # Each layer must catch every bug on its own, so neither can silently weaken behind the other.
        self.assertEqual(summary["killed_by_randomized_checks"], summary["scored"])
        self.assertEqual(summary["killed_by_fixed_checks"], summary["scored"])

    def test_randomized_check_reports_each_kind_of_discrepancy(self):
        """Each guard in run_sequence must fire for the bug it exists to catch."""
        def store(**changes):
            class Store(OracleStore):
                pass
            for name, function in changes.items():
                setattr(Store, name, function)
            return Store
        book = OracleStore.book
        listing = OracleStore.list_bookings
        many = OracleStore.book_many
        def wrong_code(self, record):
            try:
                return book(self, record)
            except ValueError as error:
                raise ValueError("CONFLICT" if str(error) != "CONFLICT" else "INVALID_INPUT")
        def wrong_result(self, record):
            result = book(self, record)
            return {**result, "start": result["start"].replace("Z", "+00:00")}
        def crashes(self, record):
            raise TypeError("boom")
        def rejects_valid(self, record):
            book(self, record)
            raise ValueError("CONFLICT")
        def mutates_input(self, records):
            result = many(self, records)
            if isinstance(records, list):
                records.append(None)
            return result
        def aliased(self, record):
            normalized = book(self, record)
            self._bookings[normalized["id"]] = normalized
            return normalized
        def aliased_listing(self):
            return [self._bookings[key] for key in sorted(self._bookings)]
        def wrong_state(self, record):
            result = book(self, record)
            stored = self._bookings[result["id"]]
            self._bookings[result["id"]] = {**stored, "room": "maple" if stored["room"] == "cedar" else "cedar"}
            return result
        cases = {
            "stored bookings differ from the oracle": ("baseline", store(book=wrong_state)),
            "wrong error code": ("baseline", store(book=wrong_code)),
            "returned": ("baseline", store(book=wrong_result)),
            "non-ValueError": ("baseline", store(book=crashes)),
            "rejected a valid operation": ("baseline", store(book=rejects_valid)),
            "mutated its input": ("batch", store(book_many=mutates_input)),
            "caller mutated inputs or returned values": ("baseline", store(book=aliased)),
            "mutated a list_bookings result": ("baseline", store(list_bookings=aliased_listing)),
        }
        for expected, (task, factory) in cases.items():
            with self.subTest(expected=expected):
                message = randomized.check(factory, task)
                self.assertIsNotNone(message)
                self.assertIn(expected, message)
        def broken_constructor():
            raise RuntimeError("cannot build")
        self.assertIn("constructor failed", randomized.run_sequence(broken_constructor, randomized.generate("baseline", 0)).message)

    def test_mutation_text_must_match_exactly_once(self):
        with self.assertRaises(ValueError):
            mutation.apply("baseline", "text that is not in the reference", "x")


if __name__ == "__main__":
    unittest.main()
