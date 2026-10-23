"""Convert a CodinGame replay JSON (gameResultRemoteService/findByGameId) into
the input of build/replay_check.

    python3 tools/replay_to_check.py replay.json | build/replay_check
"""
import gzip
import json
import math
import sys


def convert(path):
    r = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    fr = r["frames"]
    head = fr[0]["view"].split("\n")
    nums = [int(v) for v in head[3].split()]
    cps = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    states, cmds = [], []
    pending = []
    for i, f in enumerate(fr):
        if "stdout" in f:
            pending += f["stdout"].strip().split("\n")[:2]
        if not f.get("keyframe"):
            continue
        lines = f["view"].split("\n")
        off = 4 if i == 0 else 0
        st = []
        for k in range(4):
            p = lines[off + 1 + 2 * k].split()
            ang = 0.0 if p[8] == "null" else math.degrees(float(p[8])) % 360.0
            st.append(f"{float(p[0]):.0f} {float(p[1]):.0f} {p[2]} {p[3]} {ang:.9f} {p[10]}")
        states.append(st)
        if i > 0:
            if len(pending) != 4:
                break  # a player failed to answer: stop before that turn
            cmds.append(pending)
        pending = []
    out = [str(len(cps))] + [f"{x} {y}" for x, y in cps] + [str(len(cmds))]
    out += states[0]
    for t, c in enumerate(cmds):
        out += [" ".join(l.split()[:3]) for l in c]
        out += states[t + 1]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(convert(sys.argv[1]))
