"""CodinGame API helper for Mad Pod Racing: ladder battles, battle logs and test games.

Needs a token in ~/.config/codingame/token (OAuth device flow, see https://www.codingame.com/api/v1/docs/skill).
The token is read from the file, never printed.

    python3 cg_api.py battles [--submission ID]      sync battles to DATA/battles-<sub>.jsonl, print record by opponent
    python3 cg_api.py logs [--lost] [--top N]        fetch logs of synced battles into DATA/games/<id>.json
    python3 cg_api.py test BOT.cpp OPP [OPP ...] [--n K] [--seed-from GAME]   run test games, print results
    python3 cg_api.py code [--submission ID] OUT     save a submission's code
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

API = "https://www.codingame.com/api/v1"
PUZZLE = "mad-pod-racing"
DATA = os.environ.get("CG_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def token():
    with open(os.path.expanduser("~/.config/codingame/token")) as f:
        return f.read().strip()


def call(method, path, body=None, retries=6):
    url = path if path.startswith("http") else API + path
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", "Bearer " + token())
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                txt = r.read().decode()
                return r.status, (json.loads(txt) if txt.startswith(("{", "[")) else txt)
        except urllib.error.HTTPError as e:
            txt = e.read().decode()
            try:
                j = json.loads(txt)
            except ValueError:
                j = {"raw": txt}
            if e.code == 429:
                wait = float(e.headers.get("Retry-After") or j.get("retry_after_seconds") or 30)
                print(f"  rate limited ({j.get('rate_limit', {}).get('name')}), waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(wait + 1)
                continue
            if e.code >= 500 and attempt + 1 < retries:
                time.sleep(5)
                continue
            return e.code, j
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  network error {e}, retrying", file=sys.stderr)
            time.sleep(5)
    raise SystemExit(f"giving up on {method} {path}")


def sync_battles(submission=None):
    q = "?limit=100" + (f"&submission={submission}" if submission else "")
    games, before, sub = {}, None, submission
    while True:
        st, d = call("GET", f"/puzzles/{PUZZLE}/battles{q}" + (f"&before={before}" if before else ""))
        if st != 200:
            raise SystemExit(f"battles: {st} {d}")
        sub = d.get("submission_id", sub)
        for b in d.get("battles", []):
            games[b["game_id"]] = b
        before = d.get("next_before")
        if not before:
            break
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"battles-{sub}.jsonl")
    with open(path, "w") as f:
        for gid in sorted(games):
            f.write(json.dumps(games[gid]) + "\n")
    return sub, [games[g] for g in sorted(games)], path


def opponent(b):
    return next(p for p in b["players"] if not p.get("me"))


def record(battles):
    rec = {}
    for b in battles:
        if not b.get("done"):
            continue
        o = opponent(b)["nickname"]
        w, l = rec.get(o, (0, 0))
        rec[o] = (w + (b["my_position"] == 0), l + (b["my_position"] != 0))
    return rec


def fetch_game(gid):
    d = os.path.join(DATA, "games")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{gid}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    st, j = call("GET", f"/games/{gid}?format=json")
    if st != 200:
        print(f"  game {gid}: {st} {j}", file=sys.stderr)
        return None
    with open(path, "w") as f:
        json.dump(j, f)
    return j


def run_test(code, opponents=None, replay=None, game_options=None):
    body = {"language": "C++", "code": code}
    if replay:
        body["replay_game_id"] = replay
    else:
        body["opponents"] = opponents
        if game_options:
            body["game_options"] = game_options
    while True:
        st, d = call("POST", f"/puzzles/{PUZZLE}/test-games?wait=30&format=json", body)
        if st == 409 and d.get("poll_url"):
            call("GET", d["poll_url"] + ("&" if "?" in d["poll_url"] else "?") + "wait=30&format=json")
            continue
        break
    if st not in (200, 202):
        return {"status": "error", "http": st, "body": d}
    while d.get("status") == "running":
        st, d = call("GET", f"/test-games/{d['test_game_id']}?wait=30&format=json")
    if d.get("status") == "failed" and "No code machine" in str(d.get("failure")):
        time.sleep(90)
        return run_test(code, opponents, replay, game_options)
    return d


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("battles"); b.add_argument("--submission")
    lg = sp.add_parser("logs"); lg.add_argument("--submission"); lg.add_argument("--lost", action="store_true")
    lg.add_argument("--max", type=int, default=1000)
    t = sp.add_parser("test"); t.add_argument("bot"); t.add_argument("opps", nargs="*")
    t.add_argument("--n", type=int, default=1); t.add_argument("--replay", nargs="*", type=int)
    t.add_argument("--out", help="append one JSON line per game here")
    c = sp.add_parser("code"); c.add_argument("out"); c.add_argument("--submission")
    su = sp.add_parser("suite"); su.add_argument("bot"); su.add_argument("suite"); su.add_argument("out")
    su.add_argument("--tag", required=True); su.add_argument("--first", type=int, default=0); su.add_argument("--count", type=int, default=10**9)
    a = ap.parse_args()

    if a.cmd == "battles":
        sub, bt, path = sync_battles(a.submission)
        done = [x for x in bt if x.get("done")]
        won = sum(x["my_position"] == 0 for x in done)
        print(f"submission {sub}: {len(done)} finished battles, {won} won, {len(done) - won} lost -> {path}")
        for o, (w, l) in sorted(record(bt).items(), key=lambda kv: -sum(kv[1])):
            print(f"  {o:18s} {w:2d}-{l:<2d}")
    elif a.cmd == "logs":
        sub, bt, _ = sync_battles(a.submission)
        todo = [x for x in bt if x.get("done") and (not a.lost or x["my_position"] != 0)][-a.max:]
        for i, x in enumerate(todo):
            fetch_game(x["game_id"])
            if i % 20 == 19:
                print(f"  {i + 1}/{len(todo)}", file=sys.stderr)
        print(f"{len(todo)} logs in {os.path.join(DATA, 'games')}")
    elif a.cmd == "suite":
        # fixed (opponent, map, seed) list, so two versions are compared on the same games; skips games already in OUT
        code = open(a.bot).read()
        suite = json.load(open(a.suite))[a.first:a.first + a.count]
        done = set()
        if os.path.exists(a.out):
            for l in open(a.out):
                r = json.loads(l)
                if r["tag"] == a.tag:
                    done.add(r["i"])
        for job in suite:
            if job["i"] in done:
                continue
            d = run_test(code, opponents=[job["agent_id"]], game_options=job["options"])
            log = d.get("log") or {}
            if d.get("status") != "done" or not log:
                print(f"{job['i']}: {d.get('status')} {d.get('failure') or d.get('body')}", flush=True)
                continue
            me = next(ag["index"] for ag in log["agents"] if ag.get("me"))
            won = log["ranking"][me] == 0
            stderr_lines = [f.get("stderr", "") for f in log["frames"] if f.get("agent") == me]
            with open(a.out, "a") as f:
                f.write(json.dumps({"tag": a.tag, "i": job["i"], "opp": job["opp"], "won": won,
                                    "game_id": log["game_id"], "turns": len(log["frames"]) // 2}) + "\n")
            os.makedirs(os.path.join(DATA, "tests"), exist_ok=True)
            with open(os.path.join(DATA, "tests", f"{log['game_id']}.json"), "w") as f:
                json.dump(log, f)
            print(f"{a.tag} #{job['i']} vs {job['opp']}: {'WIN ' if won else 'LOSS'} game {log['game_id']}"
                  f" (test games left {d.get('limits', {}).get('test_games', {}).get('remaining')})", flush=True)
        rs = [json.loads(l) for l in open(a.out)] if os.path.exists(a.out) else []
        rs = [r for r in rs if r["tag"] == a.tag]
        print(f"{a.tag}: {sum(r['won'] for r in rs)}/{len(rs)} won")
    elif a.cmd == "code":
        if a.submission:
            st, d = call("GET", f"/puzzles/{PUZZLE}/submissions/{a.submission}")
        else:
            st, d = call("GET", f"/puzzles/{PUZZLE}/code")
        with open(a.out, "w") as f:
            f.write(d["code"])
        print(f"{len(d['code'])} bytes -> {a.out}")
    elif a.cmd == "test":
        code = open(a.bot).read()
        jobs = [("replay", r) for r in (a.replay or [])] + [("opp", o) for o in a.opps for _ in range(a.n)]
        wins = 0
        for kind, x in jobs:
            d = run_test(code, opponents=[x if not str(x).isdigit() else int(x)] if kind == "opp" else None,
                         replay=x if kind == "replay" else None)
            log = d.get("log") or {}
            if d.get("status") != "done" or not log:
                print(f"{kind} {x}: {d.get('status')} {d.get('failure') or d.get('body')}")
                continue
            me = next(ag["index"] for ag in log["agents"] if ag.get("me"))
            won = log["ranking"][me] == 0
            wins += won
            opp = [ag["name"] for ag in log["agents"] if not ag.get("me")][0]
            print(f"{kind} {x} vs {opp}: {'WIN ' if won else 'LOSS'} game {log['game_id']} turns {len(log['frames'])}"
                  f"  (test games left: {d.get('limits', {}).get('test_games', {}).get('remaining')})", flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps({"kind": kind, "x": x, "opp": opp, "won": won, "game_id": log["game_id"],
                                        "options": log.get("game_options"), "seat": me}) + "\n")
        print(f"{wins}/{len(jobs)} won")


if __name__ == "__main__":
    main()
