# Mad Pod Racing bot (Legend rules)

A C++ bot for CodinGame's [Mad Pod Racing](https://www.codingame.com/multiplayer/bot-programming/mad-pod-racing):
an exact physics engine, an evolutionary search over 6-turn plans, and a network trained to imitate
the top Legend bots, which predicts what the opponent will do and seeds the search. Also here: a
local arena, a replay downloader, the imitation trainer, and an older Nash-DQN self-play trainer.

Everything is C++17 with no dependencies (Python only for downloading and bundling). The trainers
and the bot share one forward pass and one feature function, so what is trained is what plays.

## What real games taught us

Downloading 1,564 recent games of the top 15 Legend bots and re-simulating them changed the bot:

- **Each pod has its own BOOST.** The statement says one boost is shared by both pods; every real
  game says otherwise. With that fixed, all 380,047 turns re-simulate exactly (3 games differ only
  on their final turn).
- **Top bots save their boosts.** Their first boost comes at a median of turn 45 (17% on turn 1),
  the second at a median of turn 75. A search that values only the next 6 turns burns both boosts
  at once; valuing an unused boost at 2,500 (`boostW`) turned a 39% result into 80%.
- **Top bots play a small discrete game.** In 2.5 million of their moves, rotation is always
  -18, 0 or +18 degrees and thrust always 0 or 200 (plus SHIELD and BOOST).
- **Our opponent prediction was poor.** Evolving the opponent's plan with our own evaluation
  predicted where a top bot's pods would be 3 turns later with 454 units of error, worse than a
  simple heuristic (415). A network imitating the top bots gets 256. The bot now uses that network
  as its opponent model.
- **Losing on the clock.** The loss that started this (vs reCurse, replay 908909722) was a
  timeout: our runner sat 1,100-1,900 units from a checkpoint for 100 turns while reCurse's
  blocker rammed it, and our blocker never went to reset the team clock. v3 pulled the blocker
  toward its own checkpoint once the clock dropped below 45 turns; in the arena that lost 3 clock
  battles in 4, because the blocker was typically 6,000 units away (often 10,000) and, within a
  6-turn horizon, a pod facing the wrong way cannot even reduce its distance (10 turns to turn
  round), so the pull lost to keeping the enemy runner pinned. v4 triggers on *slack* (clock minus
  the turns the trip needs), rewards heading and speed toward the checkpoint, and switches the
  enemy's progress off for the blocker while it goes: clock losses to v3 went from 25 in 200 games
  to 0. A rescue inside a mutual pin is still a concession (their runner laps while ours stays
  pinned), which is why it is only a safety net and not a win on its own.
- **A blocker must not give up its checkpoint.** v3 defended the runner's next checkpoint only
  while it was closer to it than the runner; the moment a ram overshot and the runner edged
  closer, it left for the following checkpoint and the runner walked in (that is exactly how
  striker_- and Peterpanda got free, after 45 and 70 turns). `blockStick` keeps it there until the
  runner is 1,500 units ahead.
- **How the top bots' blockers beat ours (v4).** Four v3 losses (Pedro1994, striker_-, Discourage,
  Peterpanda; replays 908941296, 908941649, 908941714, 908945000) tell one story: their blocker
  waits 700-1,100 units *in front of* our runner's checkpoint, on the side it arrives from, hits it
  every other turn (148 collisions in one game) and holds it 50-100 turns per lap, while our blocker
  sat on the checkpoint centre (a runner slips past at the 600 radius) and dawdled 2,000-8,000 units
  from the action. `blockOff` copies their standing point and doubling `blockDistW` makes ours hurry.
  A pod sitting within ~200 of the centre makes a checkpoint physically unreachable, so the
  only counters are the other pod or patience. Sending our blocker to knock the pinning pod away
  ("escort") was tried in five variants and lost 25-51% of games: it frees their runner.
- **Plan against two opinions, never against the worst case (v5).** The bot is extremely sensitive
  to how the opponent is predicted: evolving the opponent's plan with our own evaluation for just
  10% of the turn (a worst-case opponent) collapses it to 16.5% against itself, and an 8-turn
  horizon, where the network's prediction has drifted, scores 21%. But scoring every candidate
  plan against *both* the network's prediction and the simple heuristic prediction, and keeping
  the worse of the two (`robustW=1`), wins **76%** against v4 over 600 games (73.5% on random
  tracks) and 76.5% against v3 — with half the evaluations per plan. Plans that only work if the
  network is exactly right get filtered out. The exact form matters: weight 2 drops to 60%, a
  third opinion (last turn's prediction) to 66.5%. A retrained network on 1,478 fresh games was
  no more accurate (237 vs 237 units at 3 turns) and no stronger.
- **Where the top bots' runners differ while pinned** (`tools/pin_stats.py`, 22,207 pin episodes):
  they aim more than 25° away from the checkpoint 35% of the time (we 18%), coast with thrust 0
  27% (we 6%), keep a wider angle round the blocker, and go through at a median speed of 322 (we
  203); their episodes last a median 5 turns, ours 12. Encoding this in the evaluation (speed,
  angular offset, keeping off the blocker, detour seeds) did not help: 40-53% each.
- **The start.** Their blocker reaches the enemy runner within the first 10 turns in 32% of games
  (median first contact: turn 4) and boosts on turn 1 in 20%; ours never boosts at the start.
  Our runner's first checkpoint comes at turn 8 to their 11, so the race start itself is fine.
- **We over-shield.** Against those bots our pods issued SHIELD 6-10 times per 100 pod-turns,
  theirs 1-7 (each costs 3 turns of thrust). `shieldW` only prices shield turns left at the
  horizon and could not change that; `shieldUseW` (a flat cost per SHIELD) brings it to their rate
  but is exactly neutral against v3 in 200 games, so it ships off.

## Submit to CodinGame

```
make bundle            # dist/mad_pod_bot.cpp: search + imitation network (hybrid), ~95k characters
make bundle-search     # dist/mad_pod_bot_search.cpp: search only, ~50k characters
```

In the CodinGame IDE, switch the language from Python 3 to **C++** and paste the file. It is
self-contained and under the 100,000-character limit. `make bundle WEIGHTS=nets/other_weights.h`
embeds a different network; `MPR_DEFAULT_MODE` at the top of `src/bot.cpp` forces a mode.

The bot thinks for 55 ms per turn (800 ms on the first) against CodinGame's 75 ms limit; if you
ever see timeouts there, lower `budget` in `src/bot.cpp`. The hybrid bundle exceeds the limit with
comments, so `tools/bundle.py` strips them (95.3k, ~4.7k to spare); if code grows past that, use the
128-unit network instead (`make bundle WEIGHTS=nets/bc128_weights.h`, about 14k characters smaller). The CodinGame error stream shows one line per turn (time, plan
evaluations) and a `SIMCHECK` line if a pod ever ends a turn somewhere the engine did not predict.

## Layout

| Path | What it is |
| --- | --- |
| `src/engine.h` | Physics: rotation, thrust, boost, shield, continuous-time collisions, checkpoints, friction, rounding, timeouts |
| `src/search.h` | Evolutionary search over 6-turn plans for both pods, evaluation function, heuristic seeds |
| `src/bot.cpp` | CodinGame I/O, state tracking (laps, timeouts, shields, boosts), self-check, modes |
| `src/features.h` | Network inputs (whole race from one pod's frame; duel view), action sets |
| `src/nash.h` | Zero-sum matrix-game solver (regret matching+) |
| `src/nn.h` | Dense network, int8 quantisation, Z85 text encoding for embedding weights |
| `src/nnbot.h` | Plays with an embedded network: imitation policy or duel net; rollouts for the search |
| `src/weights.h` | Embedded weights (empty = pure search) |
| `tools/arena.cpp` | Local referee: plays two bot programs with the real protocol, reports win rates |
| `tools/fetch_replays.py` | Downloads top bots' recent games from CodinGame (paced, resumable) |
| `tools/replays_to_dataset.py` | Turns downloaded replays into one text dataset |
| `tools/bc.cpp` | Re-simulates every game exactly and trains the imitation network |
| `tools/predict_eval.cpp` | Scores opponent predictors on real positions, 1-6 turns ahead |
| `tools/replay_check.cpp`, `tools/replay_to_check.py` | Replays one real game through the engine |
| `tools/replay_trace.py` | Turn-by-turn story of a real game: team clocks, checkpoints, runner/blocker, commands, collisions |
| `tools/arena_stalls.py` | Per-bot stall, pin, clock-loss and shield statistics over arena replays |
| `tools/pin_stats.py` | How runners behave while an enemy sits between them and their checkpoint, from a dataset (top bots vs us) |
| `tools/arena_blocking.py` | Blocker engagement over arena replays: time near the enemy runner and the standing-point profile, to compare with the top bots' |
| `tools/train.cpp` | Nash-DQN trainer, evaluation and export |
| `tools/engine_cli.cpp`, `tools/crosscheck.py` | Diff the engine against robostac's replay-validated referee |
| `tools/viewer.html` | Open arena replays in a browser |
| `tools/bundle.py` | Inline the headers into one submittable file |
| `nets/bc160_t15*` | Imitation network (44-160-160-7), trained on the top 15 bots; embedded by default |
| `nets/bc128*` | Smaller imitation network (top 6 bots), if you need room in the bundle |
| `nets/runner.bin`, `nets/duel*` | Self-play networks from the Nash-DQN trainer (older approach) |

## Build

```
make            # bot, arena, trainers and checking tools in build/
```

## The engine is exact

`make crosscheck` builds robostac's Go referee (checked by its author against 500+ real replays)
and plays random, collision-heavy games through both. On 300 games x 150 turns x 4 pods
(180,000 pod-turns, about 6 strong collisions per game plus frequent shields) there were
**zero** differences in position, speed, angle, checkpoint, shield or boost state.

More importantly, `make check-replays` re-simulates every downloaded real game: 1,561 of 1,564
games and all 380,047 turns match, the rest differing only on a game's final turn. Headings are
kept in radians exactly as the referee keeps them, because at e.g. 270 degrees the cosine of the
two representations rounds to opposite signs and shifts a speed by 1.

## Arena

```
build/arena --games 100 --time-ms 20 "build/bot" "MPR_MODE=heuristic build/bot"
build/arena --games 200 --time-ms 20 "MPR_PARAMS=speedW=2 build/bot" "build/bot"
build/arena --games 20 --replays replays "build/bot" "build/bot_old"   # then open tools/viewer.html
```

Each seed is played twice with seats swapped. Bots receive `MPR_TIME_MS` so a game takes seconds
instead of 75 ms per turn. 15 ms here matches CodinGame: real games show a median 30-34k plan
evaluations per turn at 55 ms, the same as this machine does in 15 ms. 100-game runs swing by
±10 points between seeds, so a change needs 300+ games before its few-percent gain is believable;
`python3 tools/arena_stalls.py DIR` on `--replays DIR` shows whether it changed stalls, pins or
shield use rather than just the score. Tracks mix the 13 classic layouts (jittered, rotated, sometimes reversed)
with random 2-8 checkpoint tracks. The summary gives the win rate with a 95% interval; a change
worth a few percent needs several hundred games to show.

## Imitating the top bots

```
python3 tools/fetch_replays.py --top 6 --per-agent 120 --out data/replays          # ~30 min, paced
python3 tools/fetch_replays.py --skip 6 --top 9 --per-agent 120 --out data/replays2
python3 tools/replays_to_dataset.py data/replays data/replays2 data/games.txt
build/bc --data data/games.txt --teacher-rank 15 --hidden 160 --epochs 20 --out nets/bc160_t15   # ~35 min on 2 cores
build/predict_eval --data data/games.txt --net nets/bc160_t15.bin --horizon 3
make bundle WEIGHTS=nets/bc160_t15_weights.h
```

Every game is re-simulated with the engine (a mismatch drops the game), and every move of a bot
ranked at or above `--teacher-rank` becomes a sample: the whole race from that pod's point of
view in, its rotation (-18 / 0 / +18) and thrust kind (0 / 200 / SHIELD / BOOST) out. Samples are
mirrored left-right, and validation games are held out by game id. The network agrees with the
top bots' actual rotation 70% of the time and thrust kind 79%.

On its own the network is a poor racer (small errors compound over a race), but the search only
needs it for a few turns at a time: predicting the opponent, and proposing a plan for us to
improve on.

## Self-play training (older approach)

```
# Stage 1: a runner alone (about 10 minutes on one core)
build/train --blocker-prob 0 --steps 3000000 --eval-every 500000 --out nets/runner

# Stage 2: runner vs blocker by self-play, from the stage-1 weights (about 45 minutes on one core)
build/train --init nets/runner.bin --demo-prob 0.3 --steps 9000000 --eps-start 0.5 \
            --eps-steps 1000000 --eval-every 1000000 --out nets/duel

build/train --eval nets/duel.bin                          # evaluate
build/train --export nets/duel.bin --to src/weights.h     # embed (prints the quantised net's eval)
```

The trainer uses every core by default (`--threads N`; 1.7x faster on 2 cores here). Each
evaluation plays 40 fixed tracks, 3 laps, from the real starting grid, comparing the network with
the hand-written heuristic as runner, as blocker, and against itself, and saves a checkpoint
(`PREFIX_<N>M.bin`). Self-play strength wobbles between checkpoints, so pick the best one in the
arena rather than assuming the last is best.

How it learns: one network outputs Q[runner action][blocker action] (7 x 7). Each target solves
that matrix game at the next state for both sides' mixed strategies (strategies from the online
net, values from the target net), with 3-step returns, prioritised replay, Huber loss and Adam.
The runner earns +1 per checkpoint plus potential-based shaping toward its next checkpoint, which
speeds learning without changing the optimal policy; the blocker gets the negative. Running out
of the 100-turn timeout ends the episode with a penalty.

What the blocker needed: with pure self-play it stayed idle for 6M steps (Q-values barely depended
on its action). Two additions fixed that: 30% of blocker episodes are driven by the heuristic
blocker as demonstrations (`--demo-prob`), which off-policy Q-learning can still value, and 40%
start with the blocker waiting on the runner's path (`--ambush-prob`).

## Results

Measured locally on 2 cores. "v1" is the previous release (search only, one boost, no rescue).
Bots get 15 ms per turn unless stated; each pair of games swaps seats.

| Test | Result |
| --- | --- |
| Engine vs 1,564 real Legend games | all 380,047 turns re-simulate exactly (3 games differ on their final turn only) |
| Engine vs robostac's referee | 0 differences in 180,000 pod-turns of random collision-heavy games |
| Predicting a top bot 3 turns ahead (800 held-out positions) | network 255 units off; heuristic 395; evolved plan (v1's method) 451 |
| **v3 (shipped) vs v1** | **100-0** (95% CI 96-100%) |
| v3 vs v3 without the network | 95-5 at 15 ms; 58-2 at 40 ms |
| v3 without the network vs v1 | 80% (boost value 2,000) |
| Boost value: 0 / 2,000 / 4,000, without network, vs v1 | 39% / 80% / 79% |
| Timeout rescue off, without network, vs v1 (boost value 0) | 14% (vs 39% with rescue) |
| **v4 (shipped) vs v3** | **62.5%** on the classic layouts (125-75, 95% CI 56-69%) and **64.0%** on random tracks (128-72, CI 57-70%); 0 clock losses to v3's 25 and 18. At 40 ms: 58% (100 games) |
| v4's `blockOff` alone vs v3 | 58% pooled over 500 games on 4 seeds (54-62% per 100-200 games) |
| `blockOff` + `blockDistW` 1.2 vs v3 | 60% pooled over 300 games on known tracks; 55% over 200 on mixed |
| v4's rescue + `blockStick` vs the same bot without them | 48.5-51% (500 games): clock losses 2 vs 21, but converted into race losses |
| v3 with 40 ms vs v3 with 15 ms | 59%: search time is a weak lever |
| **v5 (shipped) vs v4** | **82%** on the classic layouts (164-36, 95% CI 76-87%) and **80%** on random tracks (160-40); **81.5%** vs v3 (163-37); 200 games each. At 40 ms: 81% (100 games). The baked build is 49% against the same settings passed as parameters (equivalence check) |
| `robustW=1` (two opponent opinions) vs v4 | **78%**, 77%, 73.5% (random tracks), 200 games each; 76.5% vs v3 |
| ... plus `blockStick=3000`, `rescueLead=10`, `rescueTurn=10`, `reactShare=0.3`, vs `robustW=1` alone | 62% (classic) and 63.5% (random tracks) |
| DEPTH 4 / 5 / 8 vs 6 | 36% / 48% / 21% |
| `oppShare=0.1` in hybrid (evolve the opponent 10% of the turn) | 16.5% |
| Retrained network (1,478 fresh games) vs the old one, both with `robustW=1` | 50-50 |
| Single weights (`speedW`, `snapP`, `angleW`, `blockAngW`, `boostW`), POP=10, pin-behaviour terms, racing line, detour seeds | 40-55% each: noise or worse |
| "Escort" (blocker hits the pod pinning our runner), 5 variants, vs v3 | 25-51% |

What made the difference, in order: the imitation network as opponent model and seed, valuing an
unused boost, and the timeout rescue (v3); then copying where the top bots' blockers stand and how
fast they get there (v4). These are wins against our own earlier bots; the next real test is the
CodinGame arena and IDE matches against the top bots. Note the noise: 100-game runs of the same
change swung between 51% and 69% across seeds, so a few-percent gain needs 300+ games to show.

## Tuning knobs

`MPR_PARAMS` overrides evaluation weights locally, e.g. `MPR_PARAMS=angleW=4,shieldW=300,oppShare=0.3`:

| Name | Default | Meaning |
| --- | --- | --- |
| `angleW` | 4 | runner: penalty per degree off its next checkpoint |
| `speedW` | 0 | runner: reward per unit of speed toward its next checkpoint |
| `blockDistW` | 1.2 | blocker: penalty per unit from its intercept point (0.6 until v4; the blocker dawdled) |
| `blockAngW` | 3 | blocker: penalty per degree off facing the enemy runner |
| `shieldW` | 300 | penalty per thrustless shield turn still owed at the horizon |
| `timeoutW` | 2500 | per turn a team's timeout is below 25 (ours: penalty, theirs: reward) |
| `oppShare` | 0.3 (0 in hybrid) | share of the turn spent evolving the opponent's plan; 0 trusts the network's prediction |
| `boostW` | 2500 | value of each BOOST still unused at the horizon |
| `robustW` | 1 | each candidate plan is also scored against the heuristic prediction of the opponent and loses `robustW` times its shortfall there (0 = trust the network alone; 1 = the worse of the two opinions; costs 2 evaluations per plan). The single biggest gain: 76% vs v4 |
| `reactShare` | 0.3 | share of the turn spent re-evolving our plan against the network's answer to the plan we first found |
| `rescueLead` | 10 | no clock rescue while our clock leads theirs by more than this: in a mutual pin, whoever breaks off first loses the race |
| `rescueSpeed`, `rescueTurn` | 450, 10 | the blocker needs `dist / rescueSpeed + rescueTurn` turns to reach its own checkpoint |
| `rescueSlack` | 10 | the rescue ramps up over the last 10 turns of slack (clock minus turns needed) and is full at zero |
| `rescueW`, `rescueAngW`, `rescueSpeedW` | 3, 6, 3 | at full rescue: per unit from the checkpoint, per degree off heading for it, per unit of speed toward it; the enemy runner's progress is switched off for the blocker meanwhile |
| `blockStick` | 3000 | blocker: keeps defending the runner's next checkpoint unless the runner is this much closer to it (1,500 in v4) |
| `snapP` | 0.125 | chance a mutation snaps to -18/0/+18 degrees or thrust 0/200 |
| `blockOff` | 1000 | blocker: waits this far in front of the checkpoint, on the enemy runner's side (0 = on the centre) |
| `shieldUseW` | 0 | flat cost per SHIELD in our plan; 600 cuts our shielding to the top bots' rate, neutral in the arena |

Bake winners into `EvalParams` in `src/search.h` before bundling.
