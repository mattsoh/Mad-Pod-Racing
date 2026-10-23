"""Unobstructed racing speed: turns each side's pods take for a checkpoint leg with no enemy pod within 2500,
compared on the same legs of the same games."""
import pickle, sys, math, collections
G = pickle.load(open(sys.argv[1], "rb"))
TOP = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
diffs = collections.defaultdict(list)
for g in G:
    if TOP and g["opp"] not in TOP:
        continue
    T = g["turns"]
    legs = {}  # (side, leg index) -> turns, for legs run without an enemy nearby
    for k in range(4):
        side = 0 if k >> 1 == g["me"] else 1
        start, clean = None, True
        for i in range(1, len(T)):
            p, q = T[i - 1]["pods"][k], T[i]["pods"][k]
            for e in range(4):
                if e >> 1 != k >> 1 and math.hypot(T[i]["pods"][e]["x"] - q["x"], T[i]["pods"][e]["y"] - q["y"]) < 2500:
                    clean = False
            if q["passed"] > p["passed"]:
                if start is not None and clean:
                    leg = q["passed"]
                    legs.setdefault((side, leg), []).append(i - start)
                start, clean = i, True
    for (side, leg), ts in legs.items():
        if side == 0 and (1, leg) in legs:
            diffs["won" if g["won"] else "lost"].append(min(ts) - min(legs[(1, leg)]))
for k, v in diffs.items():
    v.sort()
    print(f"{k}: {len(v)} paired clean legs; ours minus theirs: mean {sum(v) / len(v):+.2f} turns, median {v[len(v) // 2]:+d};"
          f" we faster {sum(x < 0 for x in v)}, equal {sum(x == 0 for x in v)}, slower {sum(x > 0 for x in v)}")
