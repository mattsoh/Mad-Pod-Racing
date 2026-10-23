"""Mean distance error of opponent predictions per horizon depth, on the held-out games of train_bc.py.
    python3 reval.py DATASET.npz GAMES_DIR EXE [EXE ...]
"""
import json, subprocess, sys, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from games import commands, track

ds, gdir, exes = sys.argv[1], sys.argv[2], sys.argv[3:]
if ds == "all":  # every game in GAMES_DIR (e.g. fresh test games no network was trained on)
    import glob, os
    test = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(gdir + "/*.json"))
else:
    game = np.load(ds)["game"]
    games = np.unique(game)
    test = sorted(np.random.RandomState(1).choice(games, max(1, int(0.15 * len(games))), replace=False))


def inp(gid):
    log = json.load(open(f"{gid_path(gid)}"))
    cps, laps = track(log)
    me = next(a["index"] for a in log["agents"] if a.get("me"))
    lines = [f"{laps} {len(cps)}"] + [f"{x} {y}" for x, y in cps] + ["0"]
    for t in commands(log):
        for tx, ty, kind, th, _ in t:
            lines.append(f"{tx} {ty} {kind if kind != 'THRUST' else th}")
    lines.append(f"me {me}")
    return "\n".join(lines) + "\n"


def gid_path(gid):
    return f"{gdir}/{gid}.json"


inputs = [inp(g) for g in test]
for exe in exes:
    def run(s):
        return subprocess.run([exe], input=s, capture_output=True, text=True).stdout
    acc = collections.defaultdict(list)
    with ThreadPoolExecutor(4) as ex:
        for out in ex.map(run, inputs):
            for l in out.split("\n"):
                if l:
                    t, d, m, e = l.split()
                    acc[(int(m), int(d))].append(float(e))
    print(exe, "games", len(inputs))
    for m, nm in ((0, "network"), (1, "heuristic")):
        print(f"  {nm:9s}", " ".join(f"d{d}:{np.mean(acc[(m, d)]):7.2f}" for d in range(1, 7)))
