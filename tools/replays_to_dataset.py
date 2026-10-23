"""Turn downloaded replays into one text dataset for build/bc and build/replay_check.

    python3 tools/replays_to_dataset.py data/replays data/replays2 data/games.txt

Per game:
    G <gameId> <checkpointCount> <x y ...> <rank of player 0> <rank of player 1> <won 0> <won 1>
    <turns>
    4 state lines "x y vx vy angleDeg next"      (before turn 1)
    then per turn: 4 command lines, then 4 state lines
Ranks are leaderboard ranks (9999 if unknown).
"""
import argparse
import glob
import gzip
import json
import math
import os


def game_lines(r, rank_of):
    fr = r["frames"]
    head = fr[0]["view"].split("\n")
    nums = [int(v) for v in head[3].split()]
    cps = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    states, cmds, pending = [], [], []
    for i, f in enumerate(fr):
        if "stdout" in f:
            pending += f["stdout"].strip().split("\n")[:2]
        if not f.get("keyframe"):
            continue
        lines = f["view"].split("\n")
        off = 4 if i == 0 else 0
        st = []
        try:
            for k in range(4):
                p = lines[off + 1 + 2 * k].split()
                ang = 0.0 if p[8] == "null" else math.degrees(float(p[8])) % 360.0
                st.append(f"{float(p[0]):.0f} {float(p[1]):.0f} {p[2]} {p[3]} {ang:.9f} {p[10]}")
        except (IndexError, ValueError):
            break
        if i > 0:
            if len(pending) != 4 or any(len(l.split()) < 3 for l in pending):
                break
            cmds.append([" ".join(l.split()[:3]) for l in pending])
        states.append(st)
        pending = []
    agents = sorted(r["agents"], key=lambda a: a["index"])
    pseudo = [a["codingamer"]["pseudo"] for a in agents]
    ranks = [rank_of.get(p, 9999) for p in pseudo]
    won = [1 if x == 0 else 0 for x in r["ranks"]]
    out = [f"G {r['gameId']} {len(cps)} " + " ".join(f"{x} {y}" for x, y in cps) + f" {ranks[0]} {ranks[1]} {won[0]} {won[1]}"]
    out.append(str(len(cmds)))
    out += states[0]
    for t, c in enumerate(cmds):
        out += c
        out += states[t + 1]
    return out, pseudo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="replay folders, then the output file")
    a = ap.parse_args()
    srcs, dst = a.paths[:-1], a.paths[-1]
    meta = json.load(open(os.path.join(srcs[0], "_leaderboard.json")))
    rank_of = {m["pseudo"]: m["rank"] for m in meta.values()}
    n = turns = 0
    files = sorted({p for s in srcs for p in glob.glob(os.path.join(s, "*.json.gz"))})
    with open(dst, "w") as f:
        for path in files:
            try:
                r = json.load(gzip.open(path, "rt"))
                lines, _ = game_lines(r, rank_of)
            except Exception as e:
                print("skip", path, e)
                continue
            f.write("\n".join(lines) + "\n")
            n += 1
            turns += int(lines[1])
    print(f"{n} games, {turns} turns -> {dst}")


if __name__ == "__main__":
    main()
