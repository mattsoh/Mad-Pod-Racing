"""Download recent Mad Pod Racing games of top Legend bots from CodinGame.

Uses the same public endpoints the website calls when you open a replay.
Requests are paced (default one per second) and already-downloaded games are
skipped, so it can be stopped and resumed.

    python3 tools/fetch_replays.py --top 6 --per-agent 120 --out data/replays
    python3 tools/fetch_replays.py --game 908909722 --out data/replays
    python3 tools/fetch_replays.py --agent 6796106 --per-agent 100 --out data/ours   # our own ladder games
"""
import argparse
import gzip
import json
import os
import time
import urllib.request

API = "https://www.codingame.com/services/"


def call(endpoint, payload, retries=3):
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API + endpoint, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # network hiccup or rate limit: back off
            if attempt + 1 == retries:
                raise
            time.sleep(5 * (attempt + 1))
            print("retry", endpoint, e)


def leaderboard():
    d = call("Leaderboards/getFilteredPuzzleLeaderboard",
             ["coders-strike-back", None, "global", {"active": False, "column": "", "filter": ""}])
    return [u for u in d["users"] if u["league"]["divisionIndex"] == 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6, help="take the top N Legend bots")
    ap.add_argument("--skip", type=int, default=0, help="...after skipping the first S")
    ap.add_argument("--per-agent", type=int, default=100)
    ap.add_argument("--game", type=int, action="append", help="fetch specific game ids instead")
    ap.add_argument("--agent", type=int, help="fetch the last --per-agent battles of this agent id (ours: 6796106)")
    ap.add_argument("--out", default="data/replays")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    games = []
    if a.game:
        games = list(a.game)
    elif a.agent:
        battles = call("gamesPlayersRankingRemoteService/findLastBattlesByAgentId", [a.agent, None])
        games = [b["gameId"] for b in battles if b.get("done")][: a.per_agent]
        print(f"agent {a.agent}: {len(games)} finished battles")
    else:
        lb = leaderboard()
        meta = {u["agentId"]: {"pseudo": u["pseudo"], "rank": u["rank"], "score": u["score"]} for u in lb}
        json.dump(meta, open(os.path.join(a.out, "_leaderboard.json"), "w"))
        for u in lb[a.skip: a.skip + a.top]:
            battles = call("gamesPlayersRankingRemoteService/findLastBattlesByAgentId", [u["agentId"], None])
            ids = [b["gameId"] for b in battles if b.get("done")][: a.per_agent]
            print(f"{u['rank']:3d} {u['pseudo']:15s} {len(ids)} games")
            games += ids
            time.sleep(a.pace)
    games = list(dict.fromkeys(games))
    have = set()
    for d in {a.out, "data/replays"}:
        if os.path.isdir(d):
            have |= {f.split(".")[0] for f in os.listdir(d) if f.endswith(".json.gz")}
    todo = [g for g in games if str(g) not in have]
    print(f"{len(games)} games, {len(todo)} to download")
    for i, g in enumerate(todo):
        try:
            r = call("gameResultRemoteService/findByGameId", [str(g), None])
        except Exception as e:
            print("skip", g, e)
            continue
        with gzip.open(os.path.join(a.out, f"{g}.json.gz"), "wt") as f:
            json.dump(r, f)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}")
        time.sleep(a.pace)
    print("done")


if __name__ == "__main__":
    main()
