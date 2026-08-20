"""Round-5 differential oracle for the Quine-McCluskey engine.

Independent brute-force reference: enumerate ALL implicants over k variables,
keep those whose covered cells lie entirely inside (positives + dont-cares),
take the maximal ones as prime implicants, and find an exact minimum-size
cover of the positives by subset enumeration. Compare mixtriad.fsqca output:
  correctness  - every returned implicant covers only allowed cells;
                 the union covers every positive minterm
  primality    - every returned implicant is a brute-force prime
  minimality   - #implicants vs the exact minimum cover size
"""
import itertools
import json
import random
import sys

sys.path.insert(0, "/home/claude/mixtriad")
from mixtriad.fsqca import quine_mccluskey, _covers  # noqa: E402


def cells(imp, k):
    free = [i for i, v in enumerate(imp) if v == "-"]
    out = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        c = list(imp)
        for f, b in zip(free, bits):
            c[f] = b
        out.append(tuple(c))
    return out


def brute_primes(pos, dc, k):
    allowed = set(pos) | set(dc)
    imps = [imp for imp in itertools.product((0, 1, "-"), repeat=k)
            if all(c in allowed for c in cells(imp, k))
            and any(c in set(pos) for c in cells(imp, k))]
    primes = []
    for a in imps:
        wider = any(a != b and all(bv == "-" or bv == av for av, bv in zip(a, b))
                    for b in imps)
        if not wider:
            primes.append(a)
    return primes


def min_cover_size(primes, pos):
    pos = list(pos)
    if not pos:
        return 0
    for size in range(1, len(primes) + 1):
        for combo in itertools.combinations(primes, size):
            if all(any(_covers(p, m) for p in combo) for m in pos):
                return size
    return len(primes)


def check_case(pos, dc, k):
    sol = quine_mccluskey(list(pos), list(dc))
    allowed = set(pos) | set(dc)
    ok_cover = all(any(_covers(p, m) for p in sol) for m in pos)
    ok_domain = all(set(cells(p, k)) <= allowed for p in sol)
    primes = brute_primes(pos, dc, k)
    ok_prime = all(p in primes for p in sol)
    mc = min_cover_size(primes, pos) if len(primes) <= 22 else None
    gap = (len(sol) - mc) if mc is not None else None
    return ok_cover, ok_domain, ok_prime, gap


def main():
    rng = random.Random(2026)
    stats = {"exhaustive_k3": 0, "sampled_k4": 0, "sampled_k5": 0,
             "cover_fail": 0, "domain_fail": 0, "prime_fail": 0,
             "minimality_gaps": {}, "checked_min_cover": 0}

    # exhaustive: all 2^8 = 256 boolean functions of 3 variables, no dont-cares
    univ3 = list(itertools.product((0, 1), repeat=3))
    for mask in range(256):
        pos = [m for i, m in enumerate(univ3) if (mask >> i) & 1]
        if not pos:
            continue
        c, d, p, g = check_case(pos, [], 3)
        stats["exhaustive_k3"] += 1
        stats["cover_fail"] += (not c)
        stats["domain_fail"] += (not d)
        stats["prime_fail"] += (not p)
        if g is not None:
            stats["checked_min_cover"] += 1
            stats["minimality_gaps"][str(g)] = stats["minimality_gaps"].get(str(g), 0) + 1

    # sampled: k=4 with random dont-cares (the parsimonious-solution regime)
    univ4 = list(itertools.product((0, 1), repeat=4))
    for _ in range(1500):
        cells4 = rng.sample(univ4, rng.randint(2, 12))
        cut = rng.randint(1, len(cells4) - 1) if len(cells4) > 1 else 1
        pos, dc = cells4[:cut], cells4[cut:]
        c, d, p, g = check_case(pos, dc, 4)
        stats["sampled_k4"] += 1
        stats["cover_fail"] += (not c)
        stats["domain_fail"] += (not d)
        stats["prime_fail"] += (not p)
        if g is not None:
            stats["checked_min_cover"] += 1
            stats["minimality_gaps"][str(g)] = stats["minimality_gaps"].get(str(g), 0) + 1

    # sampled: k=5, positives only (the conservative-solution regime)
    univ5 = list(itertools.product((0, 1), repeat=5))
    for _ in range(300):
        pos = rng.sample(univ5, rng.randint(2, 14))
        c, d, p, g = check_case(pos, [], 5)
        stats["sampled_k5"] += 1
        stats["cover_fail"] += (not c)
        stats["domain_fail"] += (not d)
        stats["prime_fail"] += (not p)
        if g is not None:
            stats["checked_min_cover"] += 1
            stats["minimality_gaps"][str(g)] = stats["minimality_gaps"].get(str(g), 0) + 1

    print(json.dumps(stats, indent=1))
    json.dump(stats, open("/home/claude/mixtriad/verify/round5/oracle.json", "w"), indent=1)


if __name__ == "__main__":
    main()
