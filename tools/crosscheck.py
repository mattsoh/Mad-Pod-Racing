"""Diff our engine against robostac's replay-validated Go referee.

Plays random collision-heavy games through both, in robostac's `-test` mode,
and reports every turn where any pod's state differs.

    python3 tools/crosscheck.py --ref ref/robostac/csbref --ours build/engine_cli --games 300
"""
import argparse
import math
import random
import subprocess


def random_track(rng):
    n = rng.randint(2, 8)
    while True:
        pts = [(rng.randint(1000, 15000), rng.randint(1000, 8000)) for _ in range(n)]
        if all(math.dist(a, b) > 2000 for i, a in enumerate(pts) for b in pts[i + 1:]):
            return pts


def script(rng, track, turns):
    """Random commands that aim pods at each other often, to force collisions."""
    lines = []
    for _ in range(turns):
        lines += ["0 0 0 0 0 0"] * 4
        for _ in range(4):
            r = rng.random()
            if r < 0.45:
                x, y = rng.choice(track)
                x += rng.randint(-1500, 1500)
                y += rng.randint(-1500, 1500)
            elif r < 0.9:
                x, y = rng.randint(-2000, 18000), rng.randint(-2000, 11000)
            else:
                x, y = 8000, 4500
            r = rng.random()
            if r < 0.05:
                tok = "SHIELD"
            elif r < 0.08:
                tok = "BOOST"
            elif r < 0.5:
                tok = "200"
            else:
                tok = str(rng.randint(0, 200))
            lines.append(f"{x} {y} {tok}")
    return lines


def run(cmd, track, lines, turns):
    full = track * 3 + [track[0]]
    inp = [str(len(full))] + [f"{x} {y}" for x, y in full] + [str(turns)] + lines
    out = subprocess.run(cmd, input="\n".join(inp) + "\n", capture_output=True, text=True, timeout=60)
    return [l.split() for l in out.stdout.strip().splitlines()]


def same(a, b):
    ints_a = [int(a[i]) for i in (0, 1, 2, 3, 5, 6, 7)]
    ints_b = [int(b[i]) for i in (0, 1, 2, 3, 5, 6, 7)]
    da = (float(a[4]) - float(b[4])) % 360.0
    return ints_a == ints_b and min(da, 360 - da) < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bad_games = 0
    first_bad = None
    pod_turns = 0
    for g in range(args.games):
        track = random_track(rng)
        lines = script(rng, track, args.turns)
        a = run([args.ref, "-test"], track, lines, args.turns)
        b = run([args.ours], track, lines, args.turns)
        n = min(len(a), len(b))
        pod_turns += n
        for i in range(n):
            if not same(a[i], b[i]):
                bad_games += 1
                if first_bad is None:
                    first_bad = (g, i // 4, i % 4, a[i], b[i])
                break
    print(f"games={args.games} pod-turns compared={pod_turns} games with a divergence={bad_games}")
    if first_bad:
        g, t, k, a, b = first_bad
        print(f"first divergence: game {g} turn {t} pod {k}\n  ref : {' '.join(a)}\n  ours: {' '.join(b)}")


if __name__ == "__main__":
    main()
