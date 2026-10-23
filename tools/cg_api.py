"""CodinGame API helper for Mad Pod Racing: our ladder battles, and test games on CodinGame's servers.

Needs a token in ~/.config/codingame/token (OAuth device flow, see https://www.codingame.com/api/v1/docs/skill;
scopes challenge:read challenge:play are enough, no submit). The token is read from the file, never printed.

    python3 tools/cg_api.py battles                       # sync the current submission's battles, record by opponent
    python3 tools/cg_api.py test --code dist/mad_pod_bot.cpp --label v5 --opponent Discourage --games 10
    python3 tools/cg_api.py test --code build/x.cpp --label x --replays losses   # re-run our lost battles, same seat/map/opponent
    python3 tools/cg_api.py summary                       # results per label; paired comparison on shared replays

Results go to data/cg/tests.jsonl (one line per test game, resumable: a (label, key) already there is skipped).
Test-game limits are about 10/min, 20/5 min, 80/hour, 200/6 hours: the runner waits whenever a window is used up.
"""
import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://www.codingame.com/api/v1"
PUZZLE = "mad-pod-racing"
OUT = "data/cg"


def token():
    return open(os.path.expanduser("~/.config/codingame/token")).read().strip()


def req(method, path, body=None, tries=6):
    for attempt in range(tries):
        r = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                   headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                d = json.loads(e.read() or b"{}")
            except Exception:
                d = {}
            if e.code == 429:
                wait = int(e.headers.get("Retry-After") or d.get("retry_after_seconds") or 60) + 2
                print(f"  rate-limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            return e.code, d
        except Exception as e:  # network hiccup
            print("  retry:", e, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{method} {path} failed")


def opponents():
    _, d = req("GET", f"/puzzles/{PUZZLE}/opponents")
    return {o["nickname"]: o["agent_id"] for o in d["opponents"]}


def battles(_args):
    os.makedirs(OUT, exist_ok=True)
    _, d = req("GET", f"/puzzles/{PUZZLE}/battles?limit=100")
    sid = d["submission_id"]
    allb = list(d["battles"])
    while d.get("next_before"):
        _, d = req("GET", f"/puzzles/{PUZZLE}/battles?limit=100&before={d['next_before']}")
        allb += d["battles"]
    path = os.path.join(OUT, f"battles-{sid}.jsonl")
    with open(path, "w") as f:
        for b in allb:
            f.write(json.dumps(b) + "\n")
    done = [b for b in allb if b["done"]]
    rec = collections.defaultdict(lambda: [0, 0])
    for b in done:
        opp = next(p["nickname"] for p in b["players"] if not p.get("me"))
        rec[opp][0 if b["my_position"] == 0 else 1] += 1
    w = sum(v[0] for v in rec.values())
    print(f"submission {sid}: {len(done)} finished battles, {w} won, {len(done) - w} lost -> {path}")
    for opp, (a, b) in sorted(rec.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print(f"  {opp:16s} {a:3d}-{b:<3d}")


def done_keys():
    path = os.path.join(OUT, "tests.jsonl")
    if not os.path.exists(path):
        return set(), []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {(r["label"], r["key"]) for r in rows}, rows


def play(body):
    """One test game; returns the JSON log or None after repeated failures."""
    for attempt in range(5):
        code, d = req("POST", f"/puzzles/{PUZZLE}/test-games?wait=30&format=json", body)
        if code == 409 and d.get("test_game_id"):  # another test game still running: wait for it, then retry
            req("GET", f"/test-games/{d['test_game_id']}?wait=30&format=json")
            continue
        if code not in (200, 202):
            print("  rejected:", code, d.get("code") or d.get("type"), d.get("hint", "")[:160], flush=True)
            return None
        lim = d.get("limits", {}).get("test_games") or {}
        while d.get("status") == "running":
            _, d = req("GET", f"/test-games/{d['test_game_id']}?wait=30&format=json")
        if d.get("status") == "done":
            if lim and lim.get("remaining", 1) <= 0:  # the tightest window is used up: wait for it before the next game
                wait = lim.get("resets_in_seconds", 60) + 2
                print(f"  window {lim.get('name')} used up, waiting {wait}s", flush=True)
                time.sleep(wait)
            return d["log"]
        print("  failed:", d.get("failure"), "- waiting 120s", flush=True)
        time.sleep(120)
    return None


def test(a):
    os.makedirs(OUT, exist_ok=True)
    code = open(a.code).read()
    if len(code) > 100000:
        sys.exit(f"{a.code} has {len(code)} characters, over CodinGame's 100,000 limit")
    have, _ = done_keys()
    jobs = []  # (key, body-extra, opponent nickname)
    if a.replays:
        ids = []
        if a.replays == "losses":
            sid = max(f for f in os.listdir(OUT) if f.startswith("battles-"))
            bs = [json.loads(l) for l in open(os.path.join(OUT, sid))]
            ids = [(b["game_id"], next(p["nickname"] for p in b["players"] if not p.get("me")))
                   for b in bs if b["done"] and b["my_position"] != 0][: a.limit]
        else:
            ids = [(int(x), "?") for x in a.replays.split(",")]
        jobs = [(f"replay:{g}", {"replay_game_id": g}, opp) for g, opp in ids]
    else:
        ids = opponents()
        for nick in a.opponent:
            if nick not in ids:
                sys.exit(f"unknown opponent {nick}")
            jobs += [(f"vs:{nick}:{i}", {"opponents": [ids[nick]]}, nick) for i in range(a.games)]
    path = os.path.join(OUT, "tests.jsonl")
    todo = [j for j in jobs if (a.label, j[0]) not in have]
    print(f"{a.label}: {len(jobs)} games, {len(todo)} to play", flush=True)
    for i, (key, extra, opp) in enumerate(todo):
        log = play(dict({"language": "C++", "code": code}, **extra))
        if log is None:
            continue
        me = next(ag["index"] for ag in log["agents"] if ag.get("me"))
        opp = next(ag["name"] for ag in log["agents"] if not ag.get("me"))
        row = dict(label=a.label, key=key, game_id=log["game_id"], opponent=opp, won=log["ranking"][me] == 0,
                   frames=len(log["frames"]), options=log.get("game_options"), t=time.strftime("%Y-%m-%d %H:%M"))
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{i + 1}/{len(todo)}] {key:28s} vs {opp:14s} {'WIN ' if row['won'] else 'loss'}", flush=True)


def summary(_a):
    _, rows = done_keys()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for lab, rs in by.items():
        w = sum(r["won"] for r in rs)
        opp = collections.Counter((r["opponent"], r["won"]) for r in rs)
        per = ", ".join(f"{o} {opp[(o, True)]}-{opp[(o, False)]}" for o in sorted({r['opponent'] for r in rs}))
        print(f"{lab:24s} {w:3d}/{len(rs):<3d} won  ({per})")
    labels = list(by)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            A = {r["key"]: r["won"] for r in by[labels[i]] if r["key"].startswith("replay:")}
            B = {r["key"]: r["won"] for r in by[labels[j]] if r["key"].startswith("replay:")}
            common = set(A) & set(B)
            if common:
                ab = sum(A[k] and not B[k] for k in common)
                ba = sum(B[k] and not A[k] for k in common)
                print(f"paired on {len(common)} replays: {labels[i]} wins {sum(A[k] for k in common)}, {labels[j]} wins "
                      f"{sum(B[k] for k in common)}; only {labels[i]} won {ab}, only {labels[j]} won {ba}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("battles")
    t = sub.add_parser("test")
    t.add_argument("--code", required=True)
    t.add_argument("--label", required=True)
    t.add_argument("--opponent", action="append", default=[])
    t.add_argument("--games", type=int, default=1, help="games per opponent")
    t.add_argument("--replays", help="'losses' (from the last battles sync) or comma-separated game ids")
    t.add_argument("--limit", type=int, default=1000, help="at most this many replays")
    sub.add_parser("summary")
    a = ap.parse_args()
    {"battles": battles, "test": test, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
