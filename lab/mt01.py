"""Replay the trusted offline mt-01 fixture: python -m lab.mt01."""
import json

from evaluator.evaluate import evaluate_factory
from lab.marker import score_marker
from lab.mutation import MUTANTS, ROOT, SOURCES, apply, load


def fixture_snapshots():
    """Compose cumulative trusted snapshots from the calibrated reference code."""
    def reference(task):
        return load(SOURCES[task].read_text(encoding="utf-8"), task)

    reschedule = reference("reschedule")

    class Availability(reschedule):
        available_rooms = reference("availability").available_rooms

    _, task, _, old, new = next(m for m in MUTANTS if m[0] == "reschedule-before-conflict")

    class BatchRegression(Availability):
        book_many = reference("batch").book_many
        reschedule = load(apply(task, old, new), "rollback-regression").reschedule

    class BatchCorrect(Availability):
        book_many = reference("batch").book_many

    return {"reschedule": reschedule, "availability": Availability,
            "batch_correct": BatchCorrect, "batch_regression": BatchRegression}


NONCOMPLIANT = ("absent", "malformed")


def event_order(noncompliance, regression, unobserved):
    """Order of first marker noncompliance vs first regression, given unobserved turns."""
    if regression is None:
        if noncompliance is not None:
            return "noncompliance_only"
        return "neither_observed" if unobserved else "neither"
    if noncompliance is not None and noncompliance < regression:
        return "noncompliance_first"
    # An unobserved turn up to and including the regression turn could hide
    # an omission before or on it.
    if any(t <= regression for t in unobserved):
        return "unknown"
    if noncompliance == regression:
        return "same_turn"
    return "regression_first"


def summarize(turns):
    noncompliance = kind = regression = marker_at_regression = None
    passed = False
    for row in turns:
        turn = row["turn"]
        if noncompliance is None and row["marker"] in NONCOMPLIANT:
            noncompliance, kind = turn, row["marker"]
        if regression is None and passed and row["invariant"] == "fail":
            regression = turn
            marker_at_regression = row["marker"]
        if row["invariant"] == "pass":
            passed = True
    unobserved = [r["turn"] for r in turns if r["marker"] == "unobserved"]
    return {"first_observed_noncompliance": noncompliance,
            "first_noncompliance_kind": kind,
            "first_observed_regression": regression,
            "marker_at_regression": marker_at_regression,
            "event_order": event_order(noncompliance, regression, unobserved)}


def replay_session(fixture, snapshots, session):
    turns = []
    for turn, entry in enumerate(session["turns"], 1):
        checks = {}
        for task, start in fixture["applies_from"].items():
            if turn >= start:
                report = evaluate_factory(snapshots[entry["snapshot"]], task)
                if report["status"] != "completed":
                    raise RuntimeError(report["errors"])
                checks.update({c["id"]: c["status"] for c in report["checks"]})
        turns.append({"turn": turn, "snapshot": entry["snapshot"],
                      "marker": score_marker(entry["response"], fixture["token"])["status"],
                      "invariant": checks.get(fixture["invariant"], "not_applicable"),
                      "checks": checks})
    return {"turns": turns, "summary": summarize(turns), "expected": session["expected"]}


def replay():
    fixture = json.loads((ROOT / "fixtures/mt-01/session.json").read_text(encoding="utf-8"))
    snapshots = fixture_snapshots()
    sessions = {name: replay_session(fixture, snapshots, session)
                for name, session in fixture["sessions"].items()}
    return {"cohort": "mt-01", "source": "scripted fixture", "sessions": sessions,
            "all_expected": all(s["summary"] == s["expected"] for s in sessions.values())}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
