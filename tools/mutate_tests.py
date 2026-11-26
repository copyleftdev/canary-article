"""Mutation testing of this repository's own unit tests. Standard library only.

Plants one small bug at a time in a target module (AST-level), runs the test
suite against it in an isolated copy of the code, and reports which bugs the
tests catch ("killed") and which they miss ("survived"). Survivors point at
behaviour no test pins down.

This is development tooling: tools/ is outside the experiment's source
fingerprint, and nothing here touches artifacts or runs agents.

    python3 tools/mutate_tests.py lab/rules.py lab/report.py
    python3 tools/mutate_tests.py lab/mt01.py --tests tests.test_mt01
    python3 tools/mutate_tests.py --all --output mutation-tests.json

Operators: comparison flips (< <=, > >=, == !=, in / not in, is / is not),
and/or swap, `not x` -> `x`, True/False swap, integer n -> n+1, + / - swap,
`return x` -> `return None`, and `raise ...` -> `pass`.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
COPY = ("lab", "evaluator", "presentation", "fixtures", "tasks", "tests", "docs", "Dockerfile")
COMPARE_SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                 ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
                 ast.Is: ast.IsNot, ast.IsNot: ast.Is}


def _is_docstring(node, parent):
    return (isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and parent.body and parent.body[0] is node)


def sites(tree):
    """Every (path-to-node, operator, description) mutation site, in source order."""
    found = []
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def skip(node):
        # Leave `if __name__ == "__main__":` blocks and docstrings alone.
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.If) and isinstance(current.test, ast.Compare) \
                    and isinstance(current.test.left, ast.Name) and current.test.left.id == "__name__":
                return True
        return False

    for node in ast.walk(tree):
        if skip(node):
            continue
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                if type(op) in COMPARE_SWAPS:
                    found.append((node, ("compare", index), f"{type(op).__name__} -> {COMPARE_SWAPS[type(op)].__name__}", line))
        elif isinstance(node, ast.BoolOp):
            found.append((node, ("boolop",), f"{type(node.op).__name__} -> {'Or' if isinstance(node.op, ast.And) else 'And'}", line))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            found.append((node, ("drop_not",), "not x -> x", line))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            found.append((node, ("bool",), f"{node.value} -> {not node.value}", line))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            found.append((node, ("int",), f"{node.value} -> {node.value + 1}", line))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            found.append((node, ("binop",), f"{type(node.op).__name__} -> {'Sub' if isinstance(node.op, ast.Add) else 'Add'}", line))
        elif isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            found.append((node, ("return_none",), "return x -> return None", line))
        elif isinstance(node, ast.Raise):
            found.append((node, ("drop_raise",), "raise -> pass", line))
    return found


class _Apply(ast.NodeTransformer):
    def __init__(self, target, operation):
        self.target, self.operation = target, operation

    def visit(self, node):
        if node is not self.target:
            return self.generic_visit(node)
        kind = self.operation[0]
        if kind == "compare":
            node.ops[self.operation[1]] = COMPARE_SWAPS[type(node.ops[self.operation[1]])]()
            return node
        if kind == "boolop":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            return node
        if kind == "drop_not":
            return node.operand
        if kind == "bool":
            return ast.copy_location(ast.Constant(not node.value), node)
        if kind == "int":
            return ast.copy_location(ast.Constant(node.value + 1), node)
        if kind == "binop":
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            return node
        if kind == "return_none":
            return ast.copy_location(ast.Return(value=ast.Constant(None)), node)
        if kind == "drop_raise":
            return ast.copy_location(ast.Pass(), node)
        raise ValueError(kind)


def mutants(path):
    """Yield (index, line, description, mutated source) for one file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for index, (node, operation, description, line) in enumerate(sites(tree)):
        clone = copy.deepcopy(tree)
        # Locate the same node in the clone by walk order.
        target = list(ast.walk(clone))[list(ast.walk(tree)).index(node)]
        mutated = ast.fix_missing_locations(_Apply(target, operation).visit(clone))
        yield index, line, description, ast.unparse(mutated)


def run_tests(workdir, tests, timeout):
    if tests:
        command = [sys.executable, "-m", "unittest", "-q", "-f", *tests]
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-q", "-f", "-s", "tests"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("CANARY_DOCKER_TESTS", None)
    try:
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "survived" if result.returncode == 0 else "killed"


_WORKDIR = None


def _init_worker(base):
    global _WORKDIR
    _WORKDIR = Path(tempfile.mkdtemp(prefix="mutate-", dir=base))
    for name in COPY:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, _WORKDIR / name, ignore=shutil.ignore_patterns("__pycache__"))
        elif source.exists():
            shutil.copy2(source, _WORKDIR / name)


def _run_one(job):
    relative, index, line, description, mutated, tests, timeout = job
    path = _WORKDIR / relative
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(mutated, encoding="utf-8")
        outcome = run_tests(_WORKDIR, tests, timeout)
    finally:
        path.write_text(original, encoding="utf-8")
    return {"file": relative, "index": index, "line": line, "mutation": description, "outcome": outcome}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="files to mutate, relative to the repo root")
    parser.add_argument("--all", action="store_true", help="every module in lab/ and evaluator/")
    parser.add_argument("--tests", nargs="*", help="unittest modules to run (default: the whole suite)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    targets = list(args.targets)
    if args.all:
        targets += [str(p.relative_to(ROOT)) for d in ("lab", "evaluator") for p in sorted((ROOT / d).glob("*.py"))
                    if p.name != "__init__.py"]
    if not targets:
        parser.error("give target files or --all")

    base = Path(tempfile.mkdtemp(prefix="mutate-tests-"))
    started = time.monotonic()
    try:
        # The suite must pass on the unmutated (re-unparsed) code first, or results mean nothing.
        _init_worker(base)
        for relative in targets:
            path = _WORKDIR / relative
            original = path.read_text(encoding="utf-8")
            path.write_text(ast.unparse(ast.parse(original)), encoding="utf-8")
        if run_tests(_WORKDIR, args.tests, args.timeout * 2) != "survived":
            raise SystemExit("Baseline failed: the test suite does not pass on the unmutated targets")
        jobs = [(relative, index, line, description, mutated, args.tests, args.timeout)
                for relative in targets for index, line, description, mutated in mutants(ROOT / relative)]
        print(f"{len(jobs)} mutants across {len(targets)} files, {args.workers} workers", file=sys.stderr)
        with concurrent.futures.ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(base,)) as pool:
            results = list(pool.map(_run_one, jobs, chunksize=1))
    finally:
        shutil.rmtree(base, ignore_errors=True)

    per_file = {}
    for result in results:
        counts = per_file.setdefault(result["file"], {"killed": 0, "survived": 0, "timeout": 0})
        counts[result["outcome"]] += 1
    for counts in per_file.values():
        total = sum(counts.values())
        counts["score"] = round((counts["killed"] + counts["timeout"]) / total, 3) if total else None
    report = {"tests": args.tests or "full suite", "duration_seconds": round(time.monotonic() - started, 1),
              "per_file": per_file,
              "survivors": [r for r in results if r["outcome"] == "survived"]}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    width = max(len(f) for f in per_file)
    print(f"{'file':{width}}  killed  survived  timeout  score")
    for name, counts in per_file.items():
        print(f"{name:{width}}  {counts['killed']:6}  {counts['survived']:8}  {counts['timeout']:7}  {counts['score']:.0%}")
    print(f"total {len(results)} mutants in {report['duration_seconds']}s")


if __name__ == "__main__":
    main()
