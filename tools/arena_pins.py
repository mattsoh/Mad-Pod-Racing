"""Pinned-runner profile per bot over arena replays (build/arena --replays DIR), to compare with real games.

    python3 tools/arena_pins.py DIR

Pinned turn (as tools/pin_stats.py): an enemy pod nearer the runner's checkpoint than the runner, within 2,500
of it, runner within 4,000 of the checkpoint. Reports the share of those turns spent shielded or reloading
after a SHIELD (top-15 bots in real games: 14%; v6 on the ladder: 29%), the runner's speed toward the
checkpoint, and episode lengths.
"""
import collections, glob, json, math, os, statistics, sys


def main():
    st = collections.defaultdict(lambda: {"pinned": 0, "shield": 0, "speed": [], "eps": [], "wins": 0, "games": 0})
    for path in glob.glob(os.path.join(sys.argv[1], "*.json")):
        r = json.load(open(path))
        cps, fr, bots = r["checkpoints"], r["frames"], r["bots"]
        for team in (0, 1):
            s = st[bots[team]]
            s["games"] += 1
            run_len = 0
            for t, f in enumerate(fr):
                P = f["pods"]
                prog = lambda p: p[6] * 50000 - math.hypot(cps[p[5]][0] - p[0], cps[p[5]][1] - p[1])
                k = 2 * team if prog(P[2 * team]) >= prog(P[2 * team + 1]) else 2 * team + 1
                p = P[k]
                cx, cy = cps[p[5]]
                dc = math.hypot(cx - p[0], cy - p[1])
                pinned = False
                if dc <= 4000:
                    for e in range(4):
                        if e >> 1 == team:
                            continue
                        q = P[e]
                        if math.hypot(q[0] - p[0], q[1] - p[1]) <= 2500 and math.hypot(cx - q[0], cy - q[1]) < dc:
                            pinned = True
                if pinned:
                    s["pinned"] += 1
                    s["shield"] += p[7] > 0
                    s["speed"].append((p[2] * (cx - p[0]) + p[3] * (cy - p[1])) / max(1.0, dc))
                    run_len += 1
                elif run_len:
                    s["eps"].append(run_len)
                    run_len = 0
    for bot, s in st.items():
        e = sorted(s["eps"]) or [0]
        print(f"{bot}: pinned {s['pinned'] / s['games']:.0f} turns/game; shielded or reloading {100 * s['shield'] / max(1, s['pinned']):.1f}%;"
              f" speed toward CP median {statistics.median(s['speed']):.0f}; episodes median {e[len(e) // 2]} p90 {e[9 * len(e) // 10]}")


if __name__ == "__main__":
    main()
