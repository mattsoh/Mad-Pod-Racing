"""Builds a behaviour-cloning dataset from battle logs: python3 build_dataset.py OUT.npz LOG.json...

Arrays: X (n, 44) features, rot (degrees), thrust, kind (0 thrust, 1 shield, 2 boost, 3 reloading after a shield: thrust is 0 whatever was asked), role (0 runner, 1 blocker),
nn (n, 7) embedded network's probabilities, heur (n, 3) heuristic rot/thrust/kind, mine (1 = our pod),
game, opp (index into names), turn, last (n, 4): last turn's rotation, thrust estimate (-1 after a seen touch), seen touch, real touch.
"""
import os, subprocess, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from games import commands, track, REPLAY
import json

DATASET = os.path.join(os.path.dirname(REPLAY), "dataset")
HIDE = os.environ.get("HIDE_ENEMY_SHIELDS", "1") == "1"  # features as our bot sees them: enemy shields unknown


def one(path):
    d = json.load(open(path))
    log = d.get("log", d)
    cps, laps = track(log)
    cmds = commands(log)
    me = next(a["index"] for a in log["agents"] if a.get("me"))
    opp = next(a["name"] for a in log["agents"] if not a.get("me"))
    inp = [f"{laps} {len(cps)}"] + [f"{x} {y}" for x, y in cps] + ["0"]
    for t in cmds:
        for tx, ty, kind, th, _ in t:
            inp.append(f"{tx} {ty} {kind if kind != 'THRUST' else th}")
    env = dict(os.environ, ME=str(me)) if HIDE else None
    out = subprocess.run([DATASET], input="\n".join(inp) + "\n", capture_output=True, text=True, env=env).stdout
    rows = []
    for l in out.split("\n"):
        if not l:
            continue
        a, f, p, h, lm = l.split("|")
        a = a.split()
        rows.append((int(a[0]), int(a[1]), int(a[2]), float(a[3]), int(a[4]), int(a[5]),
                     list(map(float, f.split())), list(map(float, p.split())), list(map(float, h.split())),
                     list(map(float, lm.split()))))
    return log["game_id"], me, opp, log["ranking"][me] == 0, rows


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, paths))
    names = sorted({r[2] for r in res})
    cols = {k: [] for k in ("X", "rot", "thrust", "kind", "role", "nn", "heur", "mine", "game", "opp", "turn", "won", "last")}
    for gid, me, opp, won, rows in res:
        for turn, k, role, rot, th, kind, f, p, h, lm in rows:
            cols["last"].append(lm)
            cols["X"].append(f); cols["rot"].append(rot); cols["thrust"].append(th); cols["kind"].append(kind)
            cols["role"].append(role); cols["nn"].append(p); cols["heur"].append(h); cols["mine"].append(k >> 1 == me)
            cols["game"].append(gid); cols["opp"].append(names.index(opp)); cols["turn"].append(turn); cols["won"].append(won)
    np.savez_compressed(out, names=np.array(names), **{k: np.array(v, dtype=np.float32 if k in ("X", "nn", "heur", "rot", "last") else np.int64)
                                                     for k, v in cols.items()})
    print(f"{len(cols['rot'])} pod-turns from {len(res)} games -> {out}")


if __name__ == "__main__":
    main()
