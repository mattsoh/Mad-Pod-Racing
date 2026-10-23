"""Blocker effectiveness: how often each side's runner is hit by the enemy blocker, and how much race it loses."""
import pickle, sys, math, collections
G = pickle.load(open(sys.argv[1], "rb"))
TOP = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None


def prog(p, cps):
    nx, ny = cps[p["next"]]
    return p["passed"] * 50000 - math.hypot(nx - p["x"], ny - p["y"])


st = collections.defaultdict(float)
for g in G:
    if g["sim_winner"] < 0 or (TOP and g["opp"] not in TOP):
        continue
    cps, T = g["track"], g["turns"]
    res = "W" if g["won"] else "L"
    for i in range(1, len(T) - 1):
        t, u = T[i], T[i + 1]
        if "hit" not in t:
            continue
        for side, nm in ((g["me"], "me"), (1 - g["me"], "op")):
            a, b = 2 * side, 2 * side + 1
            run = a if prog(t["pods"][a], cps) >= prog(t["pods"][b], cps) else b
            p0, p1 = t["pods"][run], u["pods"][run]
            gain = prog(p1, cps) - prog(p0, cps)
            if gain < -20000: gain += 0  # passed-checkpoint artifacts don't occur: progress is monotone in passes
            st[(res, nm, "turns")] += 1
            st[(res, nm, "gain")] += gain
            if t["hit"][run]:
                # who hit the runner: the nearest other pod
                others = [k for k in range(4) if k != run]
                k = min(others, key=lambda k: math.hypot(t["pods"][k]["x"] - p0["x"], t["pods"][k]["y"] - p0["y"]))
                kind = "mate" if k >> 1 == side else "enemy"
                st[(res, nm, "hit_" + kind)] += 1
                st[(res, nm, "gain_hit_" + kind)] += gain
for res in "WL":
    print("WON" if res == "W" else "LOST", "games:")
    for nm in ("me", "op"):
        n = st[(res, nm, "turns")]
        he = st[(res, nm, "hit_enemy")]
        print(f"  {nm} runner: avg progress/turn {st[(res, nm, 'gain')] / n:.0f}; hit by enemy {100 * he / n:.1f}% of turns,"
              f" progress on those turns {st[(res, nm, 'gain_hit_enemy')] / max(1, he):.0f}; hit by mate {100 * st[(res, nm, 'hit_mate')] / n:.1f}%")
