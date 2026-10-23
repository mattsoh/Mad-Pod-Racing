"""Blocker engagement over arena replays (build/arena --replays DIR), per bot:
the share of turns its blocker is within 2,500 of the enemy runner, its median speed, and its standing-point
profile: how far ahead of the enemy runner it stands (on the runner's line to its checkpoint) by the
runner's distance to that checkpoint. The top 15 Legend bots in 1,474 real games: 80-84% within 2,500,
speed 200-270, and leads of 1356 / 1609 / 2175 / 2854 / 3288 for runner distances 0-2k / 2-4k / 4-6k /
6-9k / 9k+ (positions along the line +0.64 / +0.39 / +0.26 / +0.25 / +0.22).

    python3 tools/arena_blocking.py replays/
"""
import json
import math
import os
import statistics
import sys

BINS = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 9000), (9000, 99999)]


def main():
    d = sys.argv[1]
    agg = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, f)))
        cps = r["checkpoints"]
        for team in (0, 1):
            a = agg.setdefault(r["bots"][team], dict(n=0, near=0, sp=[], prof=[], games=0, pinned=0, eps=[], vcp=[]))
            a["games"] += 1
            ks = (2 * team, 2 * team + 1)
            en = (2 - 2 * team, 3 - 2 * team)
            run = 0
            for F in r["frames"][15:]:
                P = F["pods"]

                def prog(k):
                    return P[k][6] * 1e5 - math.hypot(cps[P[k][5]][0] - P[k][0], cps[P[k][5]][1] - P[k][1])

                # this team's runner pinned: an enemy closer to its checkpoint, within 2,500 of it, runner within 4,000
                OR = max(ks, key=prog)
                ox, oy = P[OR][0], P[OR][1]
                qx, qy = cps[P[OR][5]]
                dO = math.hypot(qx - ox, qy - oy)
                pin = dO < 4000 and any(math.hypot(P[e][0] - qx, P[e][1] - qy) < dO and
                                        math.hypot(P[e][0] - ox, P[e][1] - oy) < 2500 for e in en)
                if pin:
                    run += 1
                    a["pinned"] += 1
                    a["vcp"].append((P[OR][2] * (qx - ox) + P[OR][3] * (qy - oy)) / max(1, dO))
                elif run:
                    a["eps"].append(run)
                    run = 0
                if P[ks[0]][6] == P[ks[1]][6]:
                    continue
                B = min(ks, key=prog)
                R = max(en, key=prog)
                bx, by, rx, ry = P[B][0], P[B][1], P[R][0], P[R][1]
                cx, cy = cps[P[R][5]]
                dRC = math.hypot(rx - cx, ry - cy)
                ux, uy = (cx - rx) / max(1, dRC), (cy - ry) / max(1, dRC)
                a["n"] += 1
                a["near"] += math.hypot(bx - rx, by - ry) < 2500
                a["sp"].append(math.hypot(P[B][2], P[B][3]))
                a["prof"].append((dRC, math.hypot(bx - rx, by - ry), ((bx - rx) * ux + (by - ry) * uy) / max(1, dRC)))
    for b, a in agg.items():
        print(f"  {b[:60]}")
        print(f"    blocker within 2500 of the enemy runner: {100 * a['near'] / max(1, a['n']):.0f}% of turns; median speed {statistics.median(a['sp']) if a['sp'] else 0:.0f}")
        cells = []
        for lo, hi in BINS:
            L = [s for s in a["prof"] if lo <= s[0] < hi]
            if not L:
                cells.append("n/a")
                continue
            cells.append(f"{statistics.median(x[1] for x in L):4.0f} ({statistics.median(x[2] for x in L):+.2f})")
        print("    lead ahead of the runner (position on its line) for runner->checkpoint 0-2k / 2-4k / 4-6k / 6-9k / 9k+: " + " / ".join(cells))
        eps = sorted(a["eps"])
        if eps:
            print(f"    its RUNNER pinned {a['pinned'] / a['games']:.0f} turns per game; episodes median {eps[len(eps) // 2]} p90 {eps[int(0.9 * (len(eps) - 1))]};"
                  f" speed toward the checkpoint while pinned median {statistics.median(a['vcp']):.0f}"
                  f"   (top bots vs Discourage: episodes 4 / p90 12, speed +214; v5 on the ladder vs Discourage: 8 / 78, -32)")


if __name__ == "__main__":
    main()
