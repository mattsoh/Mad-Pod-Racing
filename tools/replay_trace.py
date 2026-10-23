"""Trace a real CodinGame game turn by turn: team clocks, checkpoints, pods, commands, collisions.

    python3 tools/replay_trace.py data/losses/908909722.json.gz              # summary
    python3 tools/replay_trace.py data/losses/908909722.json.gz --table      # every turn
    python3 tools/replay_trace.py GAME --table --from 100 --to 157 --me mattsoh

Pods 0-1 belong to the first player, 2-3 to the second; "R"/"B" marks which of a team's
pods is ahead (runner) or behind (blocker). The clock is the referee's own (turns left
before the team times out), read from the replay.
"""
import argparse
import gzip
import json
import math


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def parse(r):
    fr = r["frames"]
    head = fr[0]["view"].split("\n")
    nums = [int(v) for v in head[3].split()]
    cps = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    turns, pending, hits = [], [], []
    for i, f in enumerate(fr):
        if "stdout" in f:
            pending += [(f["agentId"], l) for l in f["stdout"].strip().split("\n")[:2]]
        if not f.get("keyframe"):
            continue
        lines = f["view"].split("\n")
        off = 4 if i == 0 else 0
        pods = []
        for k in range(4):
            p = lines[off + 1 + 2 * k].split()
            pods.append(dict(x=float(p[0]), y=float(p[1]), vx=int(p[2]), vy=int(p[3]), dead=p[5] == "1",
                             ang=None if p[8] == "null" else math.degrees(float(p[8])) % 360, shield=p[9] != "0",
                             next=int(p[10]), msg=lines[off + 2 + 2 * k].strip('"')))
        clk = lines[off + 9].split()
        clock = [int(c.split(":")[1]) for c in clk]
        lap = [int(c.split(":")[0]) for c in clk]
        hits = []
        for l in lines[off + 10:]:
            v = l.split()
            if len(v) >= 9:
                hits.append((int(v[2]), int(v[5]), float(v[1]), float(v[8])))
        cmds = [None] * 4
        if i > 0:
            byAgent = {0: [], 1: []}
            for a, l in pending:
                byAgent[a].append(l)
            cmds = byAgent[0][:2] + byAgent[1][:2]
            if len(cmds) < 4:
                cmds += [None] * (4 - len(cmds))
        turns.append(dict(pods=pods, clock=clock, lap=lap, hits=hits, cmds=cmds))
        pending = []
    return cps, turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("--me", default="mattsoh", help="our CodinGame pseudo")
    ap.add_argument("--table", action="store_true", help="print every turn")
    ap.add_argument("--from", dest="t0", type=int, default=0)
    ap.add_argument("--to", dest="t1", type=int, default=10**9)
    a = ap.parse_args()

    r = load(a.replay)
    names = {ag["index"]: ag["codingamer"]["pseudo"] for ag in r["agents"]}
    me = next((i for i, n in names.items() if n == a.me), 0)
    cps, turns = parse(r)
    n = len(cps)
    laps = 3
    total = n * laps

    # Checkpoints passed per pod, derived from changes of the next checkpoint.
    passed = [0] * 4
    hist = []
    events = []
    for t, T in enumerate(turns):
        if t > 0:
            for k in range(4):
                if T["pods"][k]["next"] != turns[t - 1]["pods"][k]["next"]:
                    passed[k] += 1
                    events.append((t, k, "cp", T["pods"][k]["next"]))
        prog = []
        for k in range(4):
            p = T["pods"][k]
            cx, cy = cps[p["next"]]
            d = math.hypot(cx - p["x"], cy - p["y"])
            prog.append(passed[k] * 100000 - d)
            p["d"] = d
            p["passed"] = passed[k]
            want = math.degrees(math.atan2(cy - p["y"], cx - p["x"])) % 360
            p["off"] = None if p["ang"] is None else (want - p["ang"] + 540) % 360 - 180
        T["role"] = ["R" if prog[k] >= prog[k ^ 1] else "B" for k in range(4)]
        hist.append(list(passed))

    def team(k):
        return "us " if (k >> 1) == me else "opp"

    def podname(k):
        return f"{team(k)}{k & 1}"

    print(f"game {r.get('gameId')}: {names.get(0)} (pods 0-1) vs {names.get(1)} (pods 2-3); we are player {me} ({names.get(me)})")
    print(f"track: {n} checkpoints x {laps} laps = {total} to finish: " + " ".join(f"{i}:({x},{y})" for i, (x, y) in enumerate(cps)))
    win = min(range(2), key=lambda i: r["ranks"][i])
    print(f"winner {names.get(win)} ({'US' if win == me else 'OPPONENT'}) after {len(turns) - 1} turns"
          f" (the winner's final checkpoint ends the game, so it is not counted below)")
    for tip in r.get("tooltips") or []:
        try:
            tj = json.loads(tip)
            print(f"  tooltip turn {tj['turn'] // 2}: {tj['text'].replace('$0', names.get(0)).replace('$1', names.get(1))}")
        except Exception:
            print("  tooltip", tip)

    if a.table:
        print("\nturn clk(us,opp) | per pod: role next/passed dist speed offAngle cmd")
        for t, T in enumerate(turns):
            if t < a.t0 or t > a.t1:
                continue
            c = T["clock"]
            row = f"{t:3d} {c[me]:3d} {c[1 - me]:3d} |"
            for k in ([0, 1, 2, 3] if me == 0 else [2, 3, 0, 1]):
                p = T["pods"][k]
                off = "  -" if p["off"] is None else f"{p['off']:+4.0f}"
                cmd = T["cmds"][k]
                ck = "" if cmd is None else cmd.split()[2] if len(cmd.split()) > 2 else "?"
                ck = {"SHIELD": "SH", "BOOST": "BO"}.get(ck, ck)
                row += f" {podname(k)}{T['role'][k]} {p['next']}/{p['passed']:2d} {p['d']:5.0f} {math.hypot(p['vx'], p['vy']):4.0f} {off} {ck:>3}{'*' if p['shield'] else ' '}|"
            hs = [f"{podname(x)}-{podname(y)}@{q:.0f}" for x, y, _, q in T["hits"]]
            print(row + (" hit " + ",".join(hs) if hs else ""))

    # Summary per team.
    print()
    for side in (me, 1 - me):
        ks = [2 * side, 2 * side + 1]
        cpTurns = sorted(t for t, k, e, _ in events if e == "cp" and k in ks)
        gaps, last = [], 0
        for t in cpTurns + [len(turns) - 1]:
            gaps.append((t - last, last, t))
            last = t
        worst = max(gaps)
        minClock = min(T["clock"][side] for T in turns)
        boosts = []
        for k in ks:
            boosts.append([t for t, T in enumerate(turns) if T["cmds"][k] and "BOOST" in T["cmds"][k]])
        shields = [sum(1 for T in turns[1:] if T["cmds"][k] and "SHIELD" in T["cmds"][k]) for k in ks]
        fin = [hist[-1][k] for k in ks]
        print(f"{'US ' if side == me else 'OPP'} {names.get(side):12s} checkpoints {fin[0]}+{fin[1]} of {total} each,"
              f" min clock {minClock}, longest stretch without a team checkpoint {worst[0]} turns ({worst[1]}-{worst[2]}),"
              f" boosts at {boosts}, shields {shields}")
    pairs = {}
    for t, T in enumerate(turns):
        for x, y, _, q in T["hits"]:
            key = tuple(sorted((podname(x), podname(y))))
            pairs.setdefault(key, []).append(t)
    for key, ts in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        print(f"  collisions {key[0]}-{key[1]}: {len(ts)} (turns {ts[:12]}{'...' if len(ts) > 12 else ''})")


if __name__ == "__main__":
    main()
