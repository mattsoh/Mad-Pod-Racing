# Legend bot: CodinGame replays, opponent model and local arena

The bot that plays on CodinGame (`bots/live_41511466.cpp`, the current submission, downloaded through the
CodinGame API) and the tools used to study its battles and improve it.

## CodinGame API

Needs a token in `~/.config/codingame/token` (OAuth device flow, see
https://www.codingame.com/api/v1/docs/skill). Data goes to `$CG_DATA` (default `cg/data/`, not committed).

    python3 tools/cg_api.py battles                     # sync the current submission's battles, record by opponent
    python3 tools/cg_api.py logs [--submission ID]      # download the battle logs
    python3 tools/cg_api.py code OUT.cpp --submission ID
    python3 tools/cg_api.py suite BOT.cpp suite.json results.jsonl --tag NAME   # fixed test-game suite

`suite.json` is a fixed list of (opponent, map, seed) test games against the players ranked around us, so two
versions are compared on the same games. CodinGame runs one test game at a time (about 40 an hour).

## Rebuilding games

Battle logs hold the map and both players' commands but no positions. `src/replay.cpp` replays the commands
through the bot's own referee physics, which matches CodinGame exactly (checked on 549 battles: our pods' aim
points are 10000 units from the rebuilt positions to within 1 unit). `tools/games.py` loads a log into per-turn
states; `tools/analyze.py`, `tools/shields.py` and `tools/blocking.py` are the loss analyses.

## Opponent model

The search predicts the opponent with the embedded network (44 features -> 160 -> 160 -> 7). `src/dataset.cpp`
and `tools/build_dataset.py` turn battles into training samples (features as our bot sees them: enemy shields
hidden), `tools/train_bc.py` trains or fine-tunes it (`--init` the live network) and exports the weights blob,
`tools/embed.py` puts a blob into a bot. `src/rollout_eval.cpp` + `tools/reval.py` score a network by how far its
6-turn rollouts end up from where the opponent really went, which is what the search needs (one-step accuracy
can disagree with it). `tools/try_net.sh` does train, embed and score in one go.

## Local arena

    g++ -O2 -std=c++17 -pthread -DBOT_SOURCE='"../bots/live_41511466.cpp"' -o build/arena src/arena.cpp
    build/arena build/A build/B 200 4 maps.txt SEED     # ENV_A / ENV_B="MPR_TIME_MS=70,..." per side

Plays two bot executables with the referee physics on maps from real battles, each map from both seats.
`src/bench.cpp` measures search speed (evaluations per ms) on a recorded game.

## Results so far (October 2026)

| bot | what changed | local arena vs live | CodinGame suite (paired, seat 0) |
|---|---|---|---|
| `bots/live_41511466.cpp` | current submission, Legend #17 | - | 11/40 |
| `bots/fastx.cpp` | search ~57% faster: interpolated trig table, cast-based floor/trunc (rebuilt games identical) | 55.0% ± 3.5 (200) | not run (the rounded-angle variant: 13/40) |
| `bots/lmm_e12.cpp` | fastx + opponent model `nets/lmm_e12.txt`: the live network fine-tuned on 628 Legend games, plus the opponent's last move as input (masked after the first rollout step) | 61.5% ± 3.4 (200); 63.0% ± 3.4 vs fastx | 7/33 vs live's 10/33 on the same games: noise at this size |

Held-out 6-turn opponent prediction error (mean distance, 71 unseen games): live 67 / 175 / 305 / 449 / 603 / 768,
lmm_e12 59 / 160 / 285 / 426 / 580 / 744 (depths 1-6).

Things that did not help: whole-degree rotations in the search (47.5% vs live: the precision costs more than the
speed gains), a flat cost per SHIELD (52.5%), training the network from scratch on 549 games (worse rollouts than
the live network despite better one-step accuracy), more search time beyond 55 ms on CodinGame (turns already
overshoot by up to 17 ms there; 63 ms reached 85 ms).
