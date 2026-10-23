"""Summarise a folder of our own CodinGame games (tools/fetch_replays.py --agent 6796106 --out data/ours).

    python3 tools/our_games.py data/ours                 # one line per game + totals split by result
    python3 tools/our_games.py data/ours --me mattsoh --losses   # losses only

Per game: opponent, result and how it ended (race or team clock), turns, checkpoints of each team's
runner at the end, and for each side's blocker against the enemy runner: share of turns within 2,500,
median speed, collisions with it, and the runner hold time (median turns from arriving within 2,500 of
its checkpoint to passing it). Totals compare wins with losses so the deciding factors stand out.
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import load, parse  # noqa: E402


def analyse(path, me_name):
    r = load(path)
    names = {a["index"]: a["codingamer"]["pseudo"] for a in r["agents"]}
    me = next((i for i, n in names.items() if n == me_name), None)
    if me is None:
        return None
    cps, turns = parse(r)
    won = r["ranks"][me] == 0
    tips = " ".join(r.get("tooltips") or [])
    timeout = "in time" in tips
    passed = [0] * 4
    prev = None
    side = {s: dict(n=0, near=0, sp=[], hits=0, holds=[], arrive={}, shields=0) for s in (0, 1)}
    minclock = [100, 100]
    for t, T in enumerate(turns):
        P = T["pods"]
        if prev:
            for k in range(4):
                if P[k]["next"] != prev[k]["next"]:
                    passed[k] += 1
        hitset = {(x, y) for x, y, _, _ in T["hits"]} | {(y, x) for x, y, _, _ in T["hits"]}
        for s in (0, 1):
            minclock[s] = min(minclock[s], T["clock"][s])
            ks = (2 * s, 2 * s + 1)
            en = (2 - 2 * s, 3 - 2 * s)

            def prog(k):
                c = cps[P[k]["next"]]
                return passed[k] * 1e5 - math.hypot(c[0] - P[k]["x"], c[1] - P[k]["y"])

            B = min(ks, key=prog)
            R = max(en, key=prog)
            S = side[s]
            for k in ks:
                c = (T["cmds"][k] or "").split()
                if len(c) > 2 and c[2] == "SHIELD":
                    S["shields"] += 1
            if t < 10:
                continue
            S["n"] += 1
            S["near"] += math.hypot(P[B]["x"] - P[R]["x"], P[B]["y"] - P[R]["y"]) < 2500
            S["sp"].append(math.hypot(P[B]["vx"], P[B]["vy"]))
            S["hits"] += (B, R) in hitset
            # hold time of the enemy runner at its checkpoint
            c = cps[P[R]["next"]]
            key = (R, passed[R])
            if math.hypot(c[0] - P[R]["x"], c[1] - P[R]["y"]) < 2500:
                S["arrive"].setdefault(key, t)
            if prev and P[R]["next"] != prev[R]["next"]:
                k0 = (R, passed[R] - 1)
                if k0 in S["arrive"]:
                    S["holds"].append(t - S["arrive"][k0])
        prev = P

    def runner_cps(s):
        return max(passed[2 * s], passed[2 * s + 1])

    def blk(s):
        S = side[s]
        return dict(near=S["near"] / max(1, S["n"]), speed=statistics.median(S["sp"]) if S["sp"] else 0,
                    hits=S["hits"], hold=statistics.median(S["holds"]) if S["holds"] else 0, shields=S["shields"])

    return dict(game=os.path.basename(path).split(".")[0], opp=names[1 - me], won=won, timeout=timeout,
                turns=len(turns) - 1, ours=runner_cps(me), theirs=runner_cps(1 - me), total=len(cps) * 3,
                minclock=minclock[me], us=blk(me), them=blk(1 - me))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--me", default="mattsoh")
    ap.add_argument("--losses", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in sorted(os.listdir(a.folder)):
        if f.endswith(".json.gz") or f.endswith(".json"):
            g = analyse(os.path.join(a.folder, f), a.me)
            if g:
                rows.append(g)
    print("game       result  opponent        turns  runner CPs us/them  | OUR blocker near/speed/hits/their hold | THEIR blocker near/speed/hits/our hold | shields us/them")
    for g in rows:
        if a.losses and g["won"]:
            continue
        u, t = g["us"], g["them"]
        res = ("WIN " if g["won"] else "LOSS") + ("*" if g["timeout"] else " ")
        print(f"{g['game']}  {res}  {g['opp'][:14]:14s} {g['turns']:5d}  {g['ours']:2d}/{g['theirs']:2d} of {g['total']:2d}        |"
              f"  {100*u['near']:3.0f}% {u['speed']:4.0f} {u['hits']:3d} {u['hold']:4.0f}               |"
              f"  {100*t['near']:3.0f}% {t['speed']:4.0f} {t['hits']:3d} {t['hold']:4.0f}               | {u['shields']:3d}/{t['shields']:3d}")
    print("(* = decided by a team clock running out)")
    for label, sel in (("WINS", [g for g in rows if g["won"]]), ("LOSSES", [g for g in rows if not g["won"]])):
        if not sel:
            continue
        m = lambda f: statistics.median(f(g) for g in sel)
        print(f"{label:6s} {len(sel):3d} games: our blocker near {100*m(lambda g: g['us']['near']):.0f}%, speed {m(lambda g: g['us']['speed']):.0f}, hits {m(lambda g: g['us']['hits']):.0f}, their runner held {m(lambda g: g['us']['hold']):.0f} turns"
              f" | their blocker near {100*m(lambda g: g['them']['near']):.0f}%, hits {m(lambda g: g['them']['hits']):.0f}, our runner held {m(lambda g: g['them']['hold']):.0f} turns"
              f" | shields {m(lambda g: g['us']['shields']):.0f}/{m(lambda g: g['them']['shields']):.0f} | clock losses {sum(g['timeout'] for g in sel)}")


if __name__ == "__main__":
    main()
