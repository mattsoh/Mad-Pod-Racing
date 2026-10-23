"""Stall statistics over a directory of arena replays (build/arena --replays DIR).

    python3 tools/arena_stalls.py replays/

Per bot: wins, the mean longest stretch without a team checkpoint, the mean longest run of
turns its runner spent pinned (within 3,000 of its checkpoint, an enemy within 1,300, speed
under 300), how many games had a pin of 20+ turns or a stall of 50+ turns, and games lost on
the team clock. Use it to see whether a change fixes stalls or just wins differently.
"""
import json, math, os, sys

def analyse(path, pinDist=3000, blockDist=1300):
    r = json.load(open(path))
    cps = r["checkpoints"]; fr = r["frames"]
    out = []
    shieldTurns = [0, 0]
    for team in (0, 1):
        ks = (2 * team, 2 * team + 1)
        enemy = (2 - 2 * team, 3 - 2 * team)
        last, worst, pinned, run = 0, 0, 0, 0
        prevPassed = [fr[0]["pods"][k][6] for k in ks]
        for t, f in enumerate(fr):
            P = f["pods"]
            passed = [P[k][6] for k in ks]
            if passed != prevPassed:
                last = t
            prevPassed = passed
            worst = max(worst, t - last)
            # runner = more progress
            def prog(k):
                x, y, _, _, _, nx, ps, _ = P[k]
                return ps * 1e5 - math.hypot(cps[nx][0] - x, cps[nx][1] - y)
            rk = max(ks, key=prog)
            x, y, vx, vy, _, nx, _, _ = P[rk]
            dcp = math.hypot(cps[nx][0] - x, cps[nx][1] - y)
            near = min(math.hypot(P[e][0] - x, P[e][1] - y) for e in enemy)
            shieldTurns[team] += sum(1 for k in ks if P[k][7] > 0)
            if dcp < pinDist and near < blockDist and math.hypot(vx, vy) < 300:
                run += 1
                pinned = max(pinned, run)
            else:
                run = 0
        out.append((worst, pinned, shieldTurns[team]))
    return r["winner"], r["reason"], len(fr) - 1, out, r["bots"]

def perBot(d):
    agg = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"): continue
        w, reason, turns, out, bots = analyse(os.path.join(d, f))
        for team in (0, 1):
            a = agg.setdefault(bots[team], dict(games=0, wins=0, worst=0, pinned=0, pin20=0, stall50=0, timeouts=0, shield=0, turns=0))
            a["games"] += 1; a["wins"] += w == team; a["shield"] += out[team][2]; a["turns"] += turns
            a["worst"] += out[team][0]; a["pinned"] += out[team][1]
            a["pin20"] += out[team][1] >= 20; a["stall50"] += out[team][0] >= 50
            a["timeouts"] += reason == "checkpoint timeout" and w != team
    for b, a in agg.items():
        n = a["games"]
        print(f"  {b[:45]:45s} wins {a['wins']}/{n}  mean longest no-CP stretch {a['worst']/n:.0f}  mean longest pin {a['pinned']/n:.1f}"
              f"  pinned>=20: {a['pin20']}  no-CP>=50: {a['stall50']}  lost on clock: {a['timeouts']}  shielded turns/100: {100*a['shield']/max(1,a['turns']):.1f}")

if __name__ == "__main__":
    d = sys.argv[1]
    perBot(d)
    rows = [analyse(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    n = len(rows)
    worst = [w for _, _, _, o, _ in rows for w, _, _ in o]
    pins = [p for _, _, _, o, _ in rows for _, p, _ in o]
    print(f"{n} games, avg turns {sum(r[2] for r in rows)/n:.0f}, reasons", {k: sum(1 for r in rows if r[1] == k) for k in set(r[1] for r in rows)})
    for th in (30, 50, 80):
        print(f"  team-games with a stretch >= {th} turns without a checkpoint: {sum(w >= th for w in worst)}/{2*n}")
    for th in (10, 20, 40):
        print(f"  team-games with runner pinned (near CP, enemy <1300, slow) >= {th} turns in a row: {sum(p >= th for p in pins)}/{2*n}")

