import pickle, sys, collections, math
from games import load_game
import glob, os
D = sys.argv[1]
G = [load_game(f) for f in sorted(glob.glob(D + "/*.json"))]
pickle.dump(G, open(os.path.join(D, "..", "games.pkl"), "wb"))

def prog(p, cps):
    nx, ny = cps[p["next"]]
    return p["passed"] * 50000 - math.hypot(nx - p["x"], ny - p["y"])

for label, filt in (("all", lambda g: True), ("losses", lambda g: not g["won"]), ("wins", lambda g: g["won"])):
    st = collections.Counter()
    for g in G:
        if g["sim_winner"] < 0 or not filt(g): continue
        T, cps = g["turns"], g["track"]
        for side, name in ((g["me"], "me"), (1 - g["me"], "op")):
            for i, t in enumerate(T):
                if "cmd" not in t: continue
                a, b = 2 * side, 2 * side + 1
                run = a if prog(t["pods"][a], cps) >= prog(t["pods"][b], cps) else b
                for k in (a, b):
                    if t["cmd"][k][2] != "SHIELD": continue
                    role = "run" if k == run else "blk"
                    st[(name, role, "n")] += 1
                    if t["pods"][k]["shield"] > 0 and name == "me": st[(name, role, "while_shielded")] += 1
                    if t.get("hit", [0]*4)[k]: st[(name, role, "hit")] += 1
                st[(name, "pod_turns")] += 2
    print(label)
    for name in ("me", "op"):
        pt = st[(name, "pod_turns")]
        for role in ("run", "blk"):
            n = st[(name, role, "n")]
            print(f"  {name} {role}: {n} shields = {100*n/pt:.1f} per 100 pod-turns; landed a hit same turn {st[(name, role, 'hit')]}/{n}"
                  + (f"; re-shield while shielded {st[(name, role, 'while_shielded')]}" if name == "me" else ""))
