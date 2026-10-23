"""How runners behave while an enemy pod sits between them and their checkpoint ("pinned"), from a dataset
written by tools/replays_to_dataset.py. Compares the top bots with us.

    python3 tools/pin_stats.py data/games.txt --top 15               # sides played by a bot ranked <= 15
    python3 tools/pin_stats.py data/ours.txt --ours 908941296:0,908909722:0,908941714:1

A pinned turn: an enemy pod is closer to the runner's next checkpoint than the runner, within 2,500 of it,
and the runner is within 4,000 of the checkpoint. Consecutive pinned turns (gaps of up to 3 allowed) form an
episode; an episode "escapes" if the runner passes that checkpoint while still pinned or within 3 turns.
"""
import argparse
import math
import statistics


def read_games(path):
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].startswith("G "):
            i += 1
            continue
        h = lines[i].split()
        gid, n = int(h[1]), int(h[2])
        cps = [(float(h[3 + 2 * j]), float(h[4 + 2 * j])) for j in range(n)]
        rank = (int(h[3 + 2 * n]), int(h[4 + 2 * n]))
        turns = int(lines[i + 1])
        i += 2
        states = [[lines[i + k].split() for k in range(4)]]
        i += 4
        cmds = []
        for _ in range(turns):
            cmds.append([lines[i + k].split() for k in range(4)])
            states.append([lines[i + 4 + k].split() for k in range(4)])
            i += 8
        yield gid, cps, rank, states, cmds


def angdiff(a, b):
    return (b - a + 540) % 360 - 180


def analyse(path, sides_of):
    ep_len, escaped, n_ep = [], 0, 0
    turns = dict(n=0, vcp=[], off=[], shield=0, zero=0, boost=0, around=[], steer_out=0)
    v_pass = []
    for gid, cps, rank, states, cmds in read_games(path):
        for side in sides_of(gid, rank):
            ks = (2 * side, 2 * side + 1)
            en = (2 - 2 * side, 3 - 2 * side)
            passed = [0] * 4
            prev_next = [int(s[5]) for s in states[0]]
            cur, gap = None, 0  # current episode: dict
            for t in range(1, len(states)):
                st = states[t]
                for k in range(4):
                    nx = int(st[k][5])
                    if nx != prev_next[k]:
                        passed[k] += 1
                    prev_next[k] = nx
                # runner: more progress
                def prog(k):
                    x, y = float(st[k][0]), float(st[k][1])
                    cx, cy = cps[int(st[k][5])]
                    return passed[k] * 1e5 - math.hypot(cx - x, cy - y)
                R = max(ks, key=prog)
                x, y, vx, vy = (float(v) for v in st[R][:4])
                ang = float(st[R][4])
                nx = int(st[R][5])
                cx, cy = cps[nx]
                dR = math.hypot(cx - x, cy - y)
                pin = None
                for e in en:
                    ex, ey = float(st[e][0]), float(st[e][1])
                    if math.hypot(cx - ex, cy - ey) < dR and math.hypot(ex - x, ey - y) < 2500 and dR < 4000:
                        pin = (ex, ey)
                        break
                if cur is not None:
                    if cur["pod"] == R and cur["cp"] == nx:
                        pass
                    else:  # the runner passed its checkpoint (or roles swapped): episode over
                        cur["escaped"] = cur["cp"] != nx and cur["pod"] == R
                        ep_len.append(cur["len"]); escaped += cur["escaped"]; n_ep += 1
                        if cur["escaped"]:
                            v_pass.append(cur["last_vcp"])
                        cur = None
                if pin is None:
                    if cur is not None:
                        gap += 1
                        if gap > 3:
                            ep_len.append(cur["len"]); n_ep += 1
                            cur = None
                    continue
                gap = 0
                if cur is None:
                    cur = dict(pod=R, cp=nx, len=0, escaped=False, last_vcp=0)
                cur["len"] += 1
                # stats for this pinned turn (command issued at turn t, i.e. cmds[t] if t < len(cmds))
                ux, uy = (cx - x) / max(1, dR), (cy - y) / max(1, dR)
                vcp = vx * ux + vy * uy
                cur["last_vcp"] = vcp
                to_cp = math.degrees(math.atan2(cy - y, cx - x))
                turns["n"] += 1
                turns["vcp"].append(vcp)
                turns["off"].append(abs(angdiff(ang, to_cp)))
                ex, ey = pin
                a_r = math.atan2(y - cy, x - cx)
                a_e = math.atan2(ey - cy, ex - cx)
                turns["around"].append(abs(math.degrees((a_r - a_e + math.pi) % (2 * math.pi) - math.pi)))
                if t < len(cmds):
                    c = cmds[t][R]
                    if c[2] == "SHIELD":
                        turns["shield"] += 1
                    elif c[2] == "BOOST":
                        turns["boost"] += 1
                    elif c[2] == "0":
                        turns["zero"] += 1
                    want = math.degrees(math.atan2(float(c[1]) - y, float(c[0]) - x))
                    if abs(angdiff(to_cp, want)) > 25:
                        turns["steer_out"] += 1
            if cur is not None:
                ep_len.append(cur["len"]); n_ep += 1
    n = max(1, turns["n"])
    q = lambda L, p: sorted(L)[int(p * (len(L) - 1))] if L else float("nan")
    print(f"  pinned turns {turns['n']}, episodes {n_ep}, escaped by passing {escaped} ({100*escaped/max(1,n_ep):.0f}%),"
          f" episode length median {statistics.median(ep_len) if ep_len else 0:.0f} p90 {q(ep_len, .9):.0f}")
    print(f"  per pinned turn: speed toward CP median {q(turns['vcp'], .5):.0f} p75 {q(turns['vcp'], .75):.0f};"
          f" |angle off CP| median {q(turns['off'], .5):.0f}; angular offset from the blocker (as seen from the CP) median {q(turns['around'], .5):.0f} deg")
    print(f"  commands: SHIELD {100*turns['shield']/n:.1f}%  thrust 0 {100*turns['zero']/n:.1f}%  BOOST {100*turns['boost']/n:.1f}%"
          f"  aiming >25 deg away from the CP {100*turns['steer_out']/n:.1f}%")
    if v_pass:
        print(f"  speed toward CP on the turn of escape: median {q(v_pass, .5):.0f}, p25 {q(v_pass, .25):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--top", type=int, help="sides played by a bot ranked at or above this")
    ap.add_argument("--ours", help="gameId:sideIndex,... sides that we played")
    a = ap.parse_args()
    if a.ours:
        ours = {int(k): int(v) for k, v in (kv.split(":") for kv in a.ours.split(","))}
        print(f"OUR sides in {a.data}:")
        analyse(a.data, lambda gid, rank: [ours[gid]] if gid in ours else [])
        print(f"THEIR sides in {a.data}:")
        analyse(a.data, lambda gid, rank: [1 - ours[gid]] if gid in ours else [])
    else:
        top = a.top or 15
        print(f"sides played by bots ranked <= {top} in {a.data}:")
        analyse(a.data, lambda gid, rank: [s for s in (0, 1) if rank[s] <= top])


if __name__ == "__main__":
    main()
