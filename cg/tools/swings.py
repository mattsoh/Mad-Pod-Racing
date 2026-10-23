"""Where the race is lost: per turn, the change in (our leader's progress - their leader's progress), split by
what touched each runner that turn."""
import pickle, sys, math, collections
G = pickle.load(open(sys.argv[1], "rb"))
TOP = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None


def prog(p, cps):
    nx, ny = cps[p["next"]]
    return p["passed"] * 50000 - math.hypot(nx - p["x"], ny - p["y"])


def runner(t, side, cps):
    a, b = 2 * side, 2 * side + 1
    return a if prog(t["pods"][a], cps) >= prog(t["pods"][b], cps) else b


acc = collections.defaultdict(lambda: collections.defaultdict(float))
for g in G:
    if g["sim_winner"] < 0 or (TOP and g["opp"] not in TOP):
        continue
    cps, T, me = g["track"], g["turns"], g["me"]
    res = "won" if g["won"] else "lost"
    for i in range(1, len(T) - 1):
        t, u = T[i], T[i + 1]
        if "hit" not in t:
            continue
        rm, ro = runner(t, me, cps), runner(t, 1 - me, cps)
        if runner(u, me, cps) != rm or runner(u, 1 - me, cps) != ro:
            continue  # role swap: progress jumps between pods
        dm = prog(u["pods"][rm], cps) - prog(t["pods"][rm], cps)
        do = prog(u["pods"][ro], cps) - prog(t["pods"][ro], cps)
        key = ("our runner hit" if t["hit"][rm] else "our runner free") + " / " + ("their runner hit" if t["hit"][ro] else "their runner free")
        acc[res][key] += dm - do
        acc[res]["turns " + key] += 1
for res in ("won", "lost"):
    n = sum(1 for g in G if g["sim_winner"] >= 0 and (not TOP or g["opp"] in TOP) and (g["won"] == (res == "won")))
    print(f"{res} ({n} games): lead change per game (in units; a checkpoint is ~50000)")
    for k in sorted(k for k in acc[res] if not k.startswith("turns")):
        print(f"  {k:40s} {acc[res][k] / n:+9.0f}   over {acc[res]['turns ' + k] / n:5.1f} turns/game")
