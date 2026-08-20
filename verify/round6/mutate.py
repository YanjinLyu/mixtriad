"""Round-6 mutation harness: measure the fault-detection power of the suite.

Applies single-operator AST mutations (comparison/arith/bool swaps) to the
analytic core (fsqca / regression / data), runs the fast test subset in an
isolated copy of the repo, and records kill/survive per mutant.
"""
import ast
import copy
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/home/claude/mixtriad")
WORK = pathlib.Path("/tmp/mutwork")
OUT = ROOT / "verify/round6/mutants.jsonl"

SWAPS_CMP = {ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
SWAPS_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add}
SWAPS_BOOL = {ast.And: ast.Or, ast.Or: ast.And}

TESTS = ["tests/test_fsqca.py", "tests/test_stress.py", "tests/test_properties.py"]


def find_sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in SWAPS_CMP:
            sites.append(("cmp", node.lineno, node.col_offset))
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAPS_BIN:
            sites.append(("bin", node.lineno, node.col_offset))
        elif isinstance(node, ast.BoolOp) and type(node.op) in SWAPS_BOOL:
            sites.append(("bool", node.lineno, node.col_offset))
    return sites


def mutate_source(src, site):
    kind, lineno, col = site
    tree = ast.parse(src)

    class M(ast.NodeTransformer):
        def __init__(self):
            self.done = False

        def _hit(self, node):
            return (not self.done and node.lineno == lineno
                    and node.col_offset == col)

        def visit_Compare(self, node):
            self.generic_visit(node)
            if kind == "cmp" and self._hit(node) and type(node.ops[0]) in SWAPS_CMP:
                node.ops[0] = SWAPS_CMP[type(node.ops[0])]()
                self.done = True
            return node

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if kind == "bin" and self._hit(node) and type(node.op) in SWAPS_BIN:
                node.op = SWAPS_BIN[type(node.op)]()
                self.done = True
            return node

        def visit_BoolOp(self, node):
            self.generic_visit(node)
            if kind == "bool" and self._hit(node) and type(node.op) in SWAPS_BOOL:
                node.op = SWAPS_BOOL[type(node.op)]()
                self.done = True
            return node

    m = M()
    new = m.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new)
    return ast.unparse(new) if m.done else None


def setup_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()
    for d in ("mixtriad", "tests", "examples"):
        shutil.copytree(ROOT / d, WORK / d,
                        ignore=shutil.ignore_patterns("__pycache__", "demo_out"))
    shutil.copy(ROOT / "pyproject.toml", WORK)


def run_mutant(module, site, idx):
    src = (ROOT / "mixtriad" / module).read_text()
    mutated = mutate_source(src, site)
    if mutated is None:
        return None
    (WORK / "mixtriad" / module).write_text(mutated)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                            "-p", "no:cacheprovider"],
                           cwd=WORK, capture_output=True, text=True, timeout=90)
        killed = r.returncode != 0
    except subprocess.TimeoutExpired:
        killed = True  # infinite-loop mutants count as detected
    finally:
        (WORK / "mixtriad" / module).write_text(src)  # restore
    return {"id": idx, "module": module, "site": list(site), "killed": killed}


def main(module, start, end):
    src = (ROOT / "mixtriad" / module).read_text()
    sites = find_sites(ast.parse(src))
    # deterministic spread across the file
    step = max(1, len(sites) // 24)
    chosen = sites[::step][:24]
    setup_work()
    with open(OUT, "a") as fh:
        for i, site in enumerate(chosen[start:end], start=start):
            res = run_mutant(module, site, f"{module}:{i}")
            if res:
                fh.write(json.dumps(res) + "\n")
                print(res["id"], "KILLED" if res["killed"] else "SURVIVED", site)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
