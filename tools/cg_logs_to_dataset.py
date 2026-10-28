"""Turn battle / test-game logs from the CodinGame API (tools/cg_api.py) into data/games.txt lines for
build/bc. The logs hold both players' commands but no positions: build/cglog_states rebuilds them.

    python3 tools/cg_logs_to_dataset.py --leaderboard data/replays/_leaderboard.json data/cg/games data/cg_games.txt
"""
import argparse, glob, json, os, subprocess


def commands(log):
    per = {0: [], 1: []}
    for f in log["frames"][1:]:
        out = f.get("stdout")
        if out is None:
            continue
        lines = [l for l in out.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            per[f["agent"]].append([" ".join(l.split()[:3]) for l in lines[:2]])
    n = min(len(per[0]), len(per[1]))
    return [per[0][t] + per[1][t] for t in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True, help="_leaderboard.json from tools/fetch_replays.py (ranks by name)")
    ap.add_argument("--states", default="build/cglog_states")
    ap.add_argument("paths", nargs="+", help="folders of <gameId>.json logs, then the output file")
    a = ap.parse_args()
    rank_of = {m["pseudo"]: m["rank"] for m in json.load(open(a.leaderboard)).values()}
    srcs, dst = a.paths[:-1], a.paths[-1]
    n = 0
    with open(dst, "w") as out:
        for path in sorted({p for s in srcs for p in glob.glob(os.path.join(s, "*.json"))}):
            d = json.load(open(path))
            log = d.get("log", d)
            opts = dict(l.split("=", 1) for l in log.get("game_options", "").split("\n") if "=" in l)
            if "map" not in opts:
                continue
            v = list(map(int, opts["map"].split()))
            cps = list(zip(v[::2], v[1::2]))
            cmds = commands(log)
            inp = [str(len(cps))] + [f"{x} {y}" for x, y in cps] + [str(len(cmds))] + [c for t in cmds for c in t]
            r = subprocess.run([a.states], input="\n".join(inp) + "\n", capture_output=True, text=True)
            states = r.stdout.strip().split("\n")
            if r.returncode or len(states) != 4 * (len(cmds) + 1):
                continue
            names = [ag["name"] for ag in sorted(log["agents"], key=lambda g: g["index"])]
            ranks = [rank_of.get(x, 9999) for x in names]
            won = [1 if x == 0 else 0 for x in log["ranking"]]
            out.write(f"G {log['game_id']} {len(cps)} " + " ".join(f"{x} {y}" for x, y in cps) +
                      f" {ranks[0]} {ranks[1]} {won[0]} {won[1]}\n{len(cmds)}\n")
            out.write("\n".join(states[:4]) + "\n")
            for t, c in enumerate(cmds):
                out.write("\n".join(c) + "\n" + "\n".join(states[4 * (t + 1):4 * (t + 2)]) + "\n")
            n += 1
    print(f"{n} games -> {dst}")


if __name__ == "__main__":
    main()
