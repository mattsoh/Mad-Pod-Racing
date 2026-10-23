"""When and by which pod each bot fires its BOOSTs, over a directory of arena replays (build/arena --replays DIR).

    python3 tools/arena_boosts.py DIR

A boost is a speed change along the new heading of ~650 that a touch cannot explain (the rule of
observeLastMove in src/features.h). Compare with the top-15 bots' real games: 7% of their boosts on turns
0-2, 5% on 3-20, 14% on 21-60, 74% after; 88% by the runner.
"""
import collections, glob, json, math, os, sys


def main():
    by = collections.defaultdict(lambda: collections.Counter())
    for path in glob.glob(os.path.join(sys.argv[1], "*.json")):
        r = json.load(open(path))
        cps, fr, bots = r["checkpoints"], r["frames"], r["bots"]
        n = len(cps)

        def prog(p):
            cx, cy = cps[p[5]]
            return p[6] * 50000 - math.hypot(cx - p[0], cy - p[1])
        for t in range(1, len(fr)):
            A, B = fr[t - 1]["pods"], fr[t]["pods"]
            for k in range(4):
                p, q = B[k], A[k]
                a = math.radians(p[4])
                ax, ay = p[2] / 0.85 - q[2], p[3] / 0.85 - q[3]
                along, side = ax * math.cos(a) + ay * math.sin(a), -ax * math.sin(a) + ay * math.cos(a)
                if abs(along - 650) > 3 or abs(side) > 2.5 + 0.01 * abs(along):
                    continue
                team = k >> 1
                mate = k ^ 1
                role = "runner" if prog(q) >= prog(A[mate]) else "blocker"
                turn = t - 1
                c = by[bots[team]]
                c["n"] += 1
                c[role] += 1
                c["0-2" if turn <= 2 else "3-20" if turn <= 20 else "21-60" if turn <= 60 else "61+"] += 1
    for bot, c in by.items():
        n = max(1, c["n"])
        print(f"{bot}: {c['n']} boosts; turns 0-2 / 3-20 / 21-60 / 61+: " +
              " / ".join(f"{100 * c[b] / n:.0f}%" for b in ("0-2", "3-20", "21-60", "61+")) +
              f"; runner {100 * c['runner'] / n:.0f}%")


if __name__ == "__main__":
    main()
