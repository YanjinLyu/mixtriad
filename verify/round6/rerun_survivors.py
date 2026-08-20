"""Re-run previously surviving mutants against the strengthened suite."""
import json, sys
sys.path.insert(0, "/home/claude/mixtriad/verify/round6")
import mutate as MU
MU.TESTS = ["tests/test_fsqca.py", "tests/test_stress.py",
            "tests/test_properties.py", "tests/test_mutation_gaps.py"]

surv = [json.loads(l) for l in open("/home/claude/mixtriad/verify/round6/mutants.jsonl")
        if not json.loads(l)["killed"]]
lo, hi = int(sys.argv[1]), int(sys.argv[2])
MU.setup_work()
out = open("/home/claude/mixtriad/verify/round6/mutants_round2.jsonl", "a")
for s in surv[lo:hi]:
    res = MU.run_mutant(s["module"], tuple(s["site"]), s["id"])
    out.write(json.dumps(res) + "\n")
    print(res["id"], "KILLED" if res["killed"] else "SURVIVED", s["site"])
