"""Turns CodinGame battle logs into full per-turn states, by replaying both players' commands through the
bot's referee physics (src/replay.cpp).

    from games import load_game
    g = load_game(path)   # dict: track, me (seat), opp name, won, turns: [{to: [t0, t1], pods: [4 x dict], cmd: [4 x (tx, ty, kind, thrust, msg)]}]
"""
import json, math, os, re, subprocess

REPLAY = os.environ.get("MPR_REPLAY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "replay"))


def parse_cmd(line):
    w = line.split()
    tx, ty = int(w[0]), int(w[1])
    if w[2] == "BOOST":
        return tx, ty, "BOOST", 200, " ".join(w[3:])
    if w[2] == "SHIELD":
        return tx, ty, "SHIELD", 0, " ".join(w[3:])
    return tx, ty, "THRUST", int(w[2]), " ".join(w[3:])


def commands(log):
    """Per turn, the 4 pods' commands (player 0's pods first). Stops at the first turn a player gave none."""
    per_agent = {0: [], 1: []}
    for f in log["frames"][1:]:
        out = f.get("stdout")
        if out is None:
            continue
        lines = [l for l in out.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            per_agent[f["agent"]].append([parse_cmd(lines[0]), parse_cmd(lines[1])])
    n = min(len(per_agent[0]), len(per_agent[1]))
    return [per_agent[0][t] + per_agent[1][t] for t in range(n)]


def track(log):
    opts = dict(l.split("=", 1) for l in log["game_options"].split("\n") if "=" in l)
    v = list(map(int, opts["map"].split()))
    return list(zip(v[::2], v[1::2])), int(opts.get("laps", 3))


def simulate(cps, laps, cmds, grid):
    inp = [f"{laps} {len(cps)}"] + [f"{x} {y}" for x, y in cps] + [str(grid)]
    for t in cmds:
        for tx, ty, kind, th, _ in t:
            inp.append(f"{tx} {ty} {kind if kind != 'THRUST' else th}")
    out = subprocess.run([REPLAY], input="\n".join(inp) + "\n", capture_output=True, text=True, check=True).stdout
    turns, winner, nturns = [], None, None
    lines = out.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("T "):
            _, t, a, b = l.split()
            pods = []
            for k in range(4):
                x, y, vx, vy, ang, nxt, passed, sh, bu = lines[i + 1 + k].split()
                pods.append(dict(x=float(x), y=float(y), vx=float(vx), vy=float(vy), angle=float(ang), next=int(nxt),
                                 passed=int(passed), shield=int(sh), boost=int(bu)))
            turns.append({"to": [int(a), int(b)], "pods": pods})
            i += 5
            continue
        if l.startswith("C "):
            turns[-1]["hit"] = list(map(int, l.split()[1:]))
        if l.startswith("W "):
            _, winner, nturns = l.split()
        i += 1
    for t, c in zip(turns, cmds):
        t["cmd"] = c
    return turns, int(winner), int(nturns)


def aim_error(turns, seat):
    """How far our pods' targets are from 10000 units away (the bot always aims 10000 ahead): ~0 if the replay is right."""
    err = 0.0
    for t in turns[1:]:
        if "cmd" not in t:
            continue
        for j in range(2):
            p, c = t["pods"][2 * seat + j], t["cmd"][2 * seat + j]
            err = max(err, abs(math.hypot(c[0] - p["x"], c[1] - p["y"]) - 10000))
    return err


def load_game(path):
    d = json.load(open(path))
    log = d.get("log", d)
    cps, laps = track(log)
    cmds = commands(log)
    me = next(a["index"] for a in log["agents"] if a.get("me"))
    best = None
    for grid in (0, 1):
        turns, winner, n = simulate(cps, laps, cmds, grid)
        e = aim_error(turns, me)
        if best is None or e < best[0]:
            best = (e, grid, turns, winner, n)
    e, grid, turns, winner, n = best
    won = log["ranking"][me] == 0
    return {"game_id": log["game_id"], "track": cps, "laps": laps, "me": me, "grid": grid, "aim_err": e,
            "opp": next(a["name"] for a in log["agents"] if not a.get("me")), "won": won, "sim_winner": winner,
            "sim_turns": n, "ncmd": len(cmds), "turns": turns, "scores": log["scores"]}
