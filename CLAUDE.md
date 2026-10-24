# Mad Pod Racing bot — context for Claude Code

C++17 bot for CodinGame's Mad Pod Racing, Legend league. Owner: mattsoh (Legend #261 with an older
Python bot when this started). Goal: climb Legend. Read README.md for layout, tools and results.

## Current state (v7 in dist/mad_pod_bot.cpp, not yet on the ladder; v6 kept as dist/mad_pod_bot_v6.cpp, v5 as _v5)
- v7 = v6 + `runShieldW=500` (cost per SHIELD by our root runner). Why: on the ladder v6's pinned runner spent
  29% of pinned turns shielding or reloading (top-15 bots 14%; context of each shield the same, so it is the
  count). Arena vs v6 at 40 ms, seed 21: **63.0%** (126-74); pinned shield/reload 24.5% -> 9.4%, pinned
  turns/game 224 -> 207. Other seeds/strengths were being run in parallel cloud sessions when v7 was cut.
  Same round, vs v6 (200 games, seed 21): pinRobustW=0 28.5% (shield share unchanged: the network's own
  rams drive the shields, not the heuristic worst case), pinRobustW=0.3 48%, pinNet=1 33.5% (top-bot shield
  share, but loses), pathW=1 (runner progress along the path round enemy pods) 53.5%.

- v6 = v5 + a faster engine (cast-based floor/trunc, interpolated trig table in applyMove; replays still exact)
  + network `nets/bcl_s` (44+4 inputs: each pod's last move, masked after a rollout's first step; trained by
  bc.cpp on what our bot sees: the predicted team's shields hidden, reload turns labelled thrust 0) on 1,669
  top-20 replays + 628 of our battles. 6-turn prediction error on 71 held-out games: 678 vs bc160_t15's 737.
  Arena vs v5 at 40 ms: **59.8%** (239-161 over 2 seeds). `make bundle WEIGHTS=nets/bcl_s_weights.h` (99.7k).
- Calibration on the cloud container (4 cores): 40 ms there = CodinGame's 55 ms; run the arena with -j 2,
  since it plays both bots of a game at once.
- Lesson: the previous attempt (v5 + faster engine + a net fine-tuned on only our 628 battles) won 61.5% in
  a local arena and TIED v5 on the ladder (42.53 vs 42.54). CodinGame test games cannot separate such
  versions either: two near-identical bots disagreed on 12 of 33 identical games. Only the ladder decides.

- LADDER RESULT for v6 (submission 41554993, 220 games): **41.98, rank 42** vs v5's 42.54 (#17). By opponent
  rank: top 12 0-18 (v5 0-16), 13-25 40% (v5 35%), 26-60 61% (v5 71%), below 60 all won. Not a prediction
  problem: bcl_s predicts ranks 26-60 better than bc160_t15 (798.9 vs 868.2 units at 6 turns, 63.1 vs 81.3 at 1).
  Three arena "wins" over v5 (61.5%, 56.8%, 59.8%) have now given a tie and a loss on the ladder: the arena,
  where the opponent is our own bot, does not predict the ladder for opponent-model changes.
- What v6's real games show (tools/pin_stats.py on 220 v6 games vs 2,737 top-15 sides; `python3
  tools/arena_boosts.py DIR` for arena replays): pinned runner coasts (thrust 0) 9% of pinned turns vs top 28%,
  SHIELDs 8% vs 4.5%, BOOSTs 0.1% vs 3.3%, speed toward the checkpoint 8 vs 47, episode p90 40 vs 26. Boosts:
  we use both in 93% of games, first at median turn 15 (top: 66%, turn 41, 24% never used); 56% of ours in
  the first 20 turns and 55% by the blocker (top: 12% in the first 20, 74% after turn 60, 88% by the runner).
- Knobs tried for that (off by default; 200 games vs v6 at 40 ms): `boostBlkW` (extra value of the root
  blocker's unused boost) 4000 53% (runner share of boosts 50 -> 64%), 10000 48%; `boostEarly=2` (unused boost
  worth boostW * (1 + 2 * race left)) 48%, and the start boost barely moves (37 -> 34% on turns 0-2). The search
  finds the start boost worth more than 7,500. Neutral in the arena; untested on the ladder.
- tools/bundle.py now also strips trailing comments (lines without string literals) when over the limit.

## v5 (previously shipped; v4 kept as dist/mad_pod_bot_v4.cpp)
- Evolutionary search over 6-turn plans (src/search.h) + imitation network (nets/bc160_t15, 44-160-160-7,
  trained on 2.5M moves of the top 15 Legend bots) used as the opponent model and as a seed ("hybrid").
- v5 = v4 + robust scoring against two opponent predictions (`robustW=1`, see below) + `blockStick` 3000,
  `rescueLead` 10, `rescueTurn` 10, `reactShare` 0.3. Arena vs v4: **82%** (164-36, CI 76-87%) on the classic
  layouts, 80% on random tracks, 81% at 40 ms; 81.5% vs v3. Baselines:
  `build/v5`, `build/v4`, `build/v3` compiled from the dist files without -O.
- v4 = v3 + four blocker/clock changes in the evaluation, all from real losses: the blocker waits 1,000 units
  in front of the checkpoint (`blockOff`), hurries to its post (`blockDistW` 0.6 -> 1.2), keeps defending a
  checkpoint until the runner is 1,500 units closer (`blockStick`), and the clock rescue is slack-based with
  heading/speed terms (`rescue*`). Arena vs v3: 62.5% on the classic layouts (125-75, CI 56-69%) and
  64.0% on random tracks (128-72); 253-147 pooled; 0 clock losses to v3's 43. 58% at 40 ms (100 games).
- v3 beat the previous release 100-0 and itself without the network 95-5 (15 ms) / 58-2 (40 ms).
- Not yet tested against real top bots. The trigger for v3 was a loss on time to reCurse (replay 908909722);
  v4's triggers were the four losses listed below.

## Facts verified against 1,564 real games (all 380,047 turns re-simulate exactly)
- Each pod has its OWN boost (the statement says shared — it is wrong). `g_sharedBoost = false`.
- Headings must be kept in radians like the referee (`Pod::rad`), or speeds drift by 1 at e.g. 270°.
- Top bots only use rotation -18/0/+18 and thrust 0/200 (+ SHIELD, BOOST), and save boosts
  (first at median turn 45). `boostW = 2500` values an unused boost; 0 lost badly.
- Our old evolved opponent prediction was worse than a heuristic; the imitation net halves the error
  (tools/predict_eval: 255 vs 451 units, 3 turns ahead).
- The four v3 losses (908941296 Pedro1994, 908941649 striker_-, 908941714 Discourage, 908945000 Peterpanda;
  all in data/losses/, all re-simulate exactly) share one cause: their blocker stands 700-1,100 units in
  FRONT of our runner's checkpoint on its arrival side and holds it 50-100 turns per lap; ours sat on the
  centre and dawdled 2-8k units away. A pod within ~200 of the centre makes a checkpoint unreachable.
  908941786 (daaskare) is a win, not a loss. Top bots issue SHIELD 1-7 times per 100 pod-turns; we did 6-10.

## Hard constraints
- Submission is ONE file, max 100,000 characters. dist/mad_pod_bot.cpp is 95,258 with comments stripped
  (tools/bundle.py strips them automatically once the file would exceed the limit) — ~4.7k to spare;
  remove dead knobs before adding code. If that is not enough, `make bundle WEIGHTS=nets/bc128_weights.h`
  (~14k smaller).
- 75 ms per turn on CodinGame; bot budget is 55 ms (`budget` in src/bot.cpp). First turn 1000 ms.
- CodinGame's machine is ~3.5x slower than this one: real games show a median 30-34k evals/turn (min 14k)
  at 55 ms, the same as the local arena at `--time-ms 15`. So 15 ms locally is the right calibration.
- CodinGame compiles without -O; `#pragma GCC optimize` at the top of src/bot.cpp does the work.

## Workflow
- `make` builds everything into build/. `make bundle` / `make bundle-search` write dist/.
- Test every change in the arena with enough games (CI is printed):
  `build/arena --games 100 --time-ms 15 "build/new" "build/old"` (seats swap; MPR_MODE / MPR_PARAMS env vars).
  100-game runs swing ±10 points between `--seed` values: the same change scored 69%, 61% and 51% on three
  seeds. Pool 300+ games over several seeds before believing a few-percent gain. `--replays DIR` then
  `python3 tools/arena_stalls.py DIR` shows stalls / pins / clock losses / shield use per bot.
- Keep a compiled copy of the shipped bundle (e.g. `g++ -std=c++17 -o build/v3 dist/mad_pod_bot.cpp`, no -O,
  as CodinGame compiles) as the arena baseline; the repo is not under git.
- Downloaded replays live in data/ (gitignored): `tools/fetch_replays.py` (paced 1 req/s),
  `tools/replays_to_dataset.py`, `build/bc` to retrain, `build/predict_eval` to score predictors.
- Analyse a real loss: `python3 tools/fetch_replays.py --game <id> --out data/losses`, check the engine
  with `python3 tools/replay_to_check.py data/losses/<id>.json.gz | build/replay_check`, then
  `python3 tools/replay_trace.py data/losses/<id>.json.gz [--table --from A --to B]` (clocks, checkpoints,
  runner/blocker, commands, collisions). One game turn = two replay frames; tooltips give frame/2.
- Don't `pkill -f` with a pattern that appears in the same shell command — it kills the shell.

## Tried and failed (don't repeat without a new angle)
- "Escort": blocker goes to knock away the pod pinning our runner. Five variants (with/without a push term,
  range-limited, after 3 pinned turns): 25-51% vs v3. It frees their runner; blocking better was the fix.
- Relaxed runner angle penalty while pinned (`pinAngleW`): 56%, same as without. Removed.
- `shieldW` 800: no change in shield use or score. `shieldUseW=600` (flat cost per SHIELD) cuts shield use
  3.5x to the top bots' rate but is exactly 50-50 against the same bot without it; shipped off (knob kept).
- Clock rescue as a distance-only pull (slack-based trigger, 2/unit): 49.5% vs v3, clock losses unchanged.
  Inside a 6-turn horizon a blocker facing away cannot reduce its distance (10 turns just to turn), so the
  pull lost to keeping the enemy runner pinned. Heading + speed terms with the enemy-progress term switched
  off (weighted by a per-turn constant from the root, never per plan) fixed it: clock losses 16-25 -> 0-2.
  But a rescue in a MUTUAL pin is a concession (their runner laps while ours stays pinned): against v3/cand2
  it converted clock losses into race losses 1:1 (48.5-51%). Kept because against the top bots our blocker
  was not pinning anyone anyway (Pedro1994 game: our clock reached 4), so there it is pure safety.
- More search time is a weak lever: v3 at 40 ms vs v3 at 15 ms is only 59%, and still loses on the clock.
  Don't spend effort on engine speed before strategy.
- Single-weight A/B tests vs v4 (200 games each): `speedW=0.5` 49%, `snapP=0.3` 46.5%, `angleW=2` 47%,
  `blockAngW=1` 43%. v4's weights sit at a local optimum; tweaking them is noise or worse.
- Detour seed plans for a pinned runner (`detourOff=1000`, two extra seeds aiming past the blocker): 49.5%.
  The search can already find such plans; the evaluation decides against them.
- Racing line (`nextAngW`: align with the following checkpoint while nearing this one): 55.5% on one seed,
  49.5% on another, 48.5% at double weight. Noise. The network seed already gives the racing line.
- Horizon: `-DMPR_DEPTH=8` scores **21%** vs v4 (200 games) — far worse than the 25% fewer evaluations
  explain. The network's opponent prediction degrades with horizon, so plans judged at 8 turns are judged
  against fiction. DEPTH 5: 48%, DEPTH 4: 36%. Six is the optimum for this predictor; leave it.
- `oppShare=0.1` in hybrid (evolve the opponent's plan for 10% of the turn, seeded by the network): **16.5%**.
  An adversarially evolved opponent is a terrible predictor and the bot is extremely sensitive to prediction
  quality. Corollaries: never plan against a worst-case opponent; the network's accuracy is a high-leverage
  target (retraining on fresh games is the cheap way to raise it).
- A second machine (ssh/rsync to another host) is blocked by the Claude Code auto-mode classifier; add a
  Bash permission rule for it first if you want arena batches to run there.

- On v5 (200 games each, vs v5): `shieldUseW=600` alone **38.5%**, with `blockLead=2500` 40%. The flat
  shield cost that was 50-50 on v3 now hurts: robust scoring leans on shields to hedge between the two
  predictions, and taking them away costs more than the wasted turns. Do not retry; shield use is not
  the lever, whatever the real replays suggest.
- `blockLead` (blocker never stands more than N ahead of the runner on its line): 2500 51.5%, 3000 47.5%,
  3500 53% vs v5 — neutral in the arena, as expected (the arena's runners stay near the blocker anyway), and
  the standing profile moves to the top bots' (3000: 1281/1473/2223/2428/2428 vs top 1356/1609/2175/2854/
  3288). 3500 on random tracks 51.5%; pooled over 800 games 50.9%. Default off (1e9) = v5 behaviour (checked:
  the default build is 54% vs v5 in 200 games). dist/mad_pod_bot.cpp was bundled BEFORE this knob existed
  (src now has it, off), so `make bundle` gives a slightly larger file (95.6k) with the same behaviour.
  Unproven in real games; test it there before turning it on.

## The big one (v5 candidate): robust scoring against two opponent predictions
- `robustW=1`: each candidate plan is scored against the network's prediction AND the heuristic prediction
  of the opponent, and gets the worse of the two (`Search::scoreRobust`). vs v4: **78%** (156-44, CI 72-83%);
  `robustW=0.5`: 71%. Costs 2 evaluations per plan and still wins by 28 points. Plans that only work if the
  network is exactly right are filtered out; our stalls dropped too (no-CP>=50: 166 vs 190).
  Not the same as P3's adversarial opponent (16.5%): the heuristic is a plausible second opinion, not a
  worst case over all plans. Confirmed: 73.5% on random tracks, 77% on a second seed, 76.5% vs v3
  (pooled 76% over 600 games vs v4). The exact form matters: `robustW=2` 60.5%, a third prediction
  (`robustN=2`, last turn's) 66.5%.
- On top of robustW=1 (200 games each, vs the same bot with robustW=1 only): `blockStick=3000` 57.5%,
  `rescueLead=10` 56%, `rescueTurn=10` 55.5%, `reactShare=0.3` 55%, `blockAngW=6` 54%, `pinAvoidW=1` 53%,
  `boostW=5000` 49%, `pinSpeedW=1` 47.5%, `pinAroundW=10` 40%, POP=10 51%. The pin-behaviour terms do not
  help (the search is not the problem there). The first four TOGETHER: 62% (classic) and 63.5% (random
  tracks), 200 games each — real, ~3 points apiece. All baked into v5.
- Escort, third time, on top of robustW (blocker goes to hit the pod pinning our runner): gated on the enemy
  runner leading by >= 2 checkpoints 50%; always 32%; leading by >= 1 with a push term 42%. Our long stalls
  drop (172 -> 134 per 200 games) but their freed runner wins the race. Gated on their runner being far
  from its checkpoint (`escortFree=3500`, "leaving the post is cheap"): 44%; with the behind gate too: 43%.
  Eight variants over three rounds, none above 50%. Dead; don't try again.
- Start seeds (the heuristic plan with a turn-1 BOOST for either pod, first 8 turns): 50%. The search
  already finds whatever the start offers.
- "Dual runner" (enemy leads by 2 checkpoints and our runner is pinned -> the blocker races its own
  checkpoints, same terms as a full rescue): 41%. Together with the escort results: every "both pods
  cooperate" scheme frees the enemy runner for more than it gains. Roles are already dynamic by
  construction (whichever pod has more progress at the horizon is the runner in the evaluation).
- Retrained network on 1,478 fresh games (`nets/bc160_v2`): identical prediction error (237.2 vs 237.3
  units at 3 turns), 50-50 against the old one with robustW, 40.5% without. Keep `bc160_t15`.

## Ideas not yet tried
- `python3 tools/pin_stats.py data/ours.txt --ours <gameId:side,...>` (or `data/games.txt --top 15`) compares
  runners during pins. 1,478 top-bot games (22,207 pin episodes): their runners aim >25° away from the
  checkpoint 35% of pinned turns (us 18% in our 5 real games), sit 23° round the blocker as seen from the
  checkpoint (us 18°), coast with thrust 0 27% (us 6%) and pass at a median speed of 322 (us 203); their
  episodes last a median 5 turns, ours 12. They keep off the blocker, reposition to the side, wait for the
  gap and go through fast; we push against it.
- The start (1,478 top-bot games): their blocker passes checkpoint 1 within 20 turns in only 42% of games
  (its first checkpoint comes at median turn 49, ~2 per game) and makes contact with the enemy runner in
  the first 10 turns in 32% of games (median first contact turn 4); 20% of their blockers BOOST on turn 1,
  27% of the early contacts follow such a boost. Runners boost on turn 1 in 28%. Ours: blocker never
  boosts at the start, early contact 1 game in 5; our runner's first checkpoint comes at turn 8 vs their 11,
  so the race start itself is fine — the missing piece is the start ram (`startSeeds`).
- Where top-bot blockers stand (1,474 games, by the enemy runner's distance to its checkpoint): they stay
  AHEAD OF THE RUNNER on its line to the checkpoint, not at the checkpoint — median 3,300 units ahead
  when the runner is 9k+ away, 2,850 at 6-9k, 2,175 at 4-6k, 1,600 at 2-4k, and ~1,000 in front of the
  checkpoint when the runner is within 2k (position along the runner->checkpoint line 0.22 -> 0.64);
  median speed 200-270, i.e. drifting ahead, not commuting. v5's blocker went straight to a post 1,000
  in front of the checkpoint: in the four v5 real games it was within 2,500 of the enemy runner 41-55%
  of turns (theirs 80-84%) and hit it 12-30 times a game (theirs 59-84). `blockLead` is the fix to test.
  The v5 real games (909092738, 909092737 Discourage; 909081160 sffsefsdfaf — losses; 909081259 Peterpanda —
  win; all in data/losses/, all re-simulate exactly) also show us shielding 31-65 times a game against
  their 11-17: our blocker SHIELDs 7-10% of turns (theirs 2-6%) and in one stretch sat idle at speed 6,
  shielding, 4,000 from everything. robustW raised shield use (arena: 27 -> 33 shielded turns per 100);
  `shieldUseW` is being retested on v5.
  Standing-point profile (blocker's lead ahead of the runner, by the runner's distance to its checkpoint,
  `python3 tools/arena_blocking.py DIR`): top bots 1356 / 1609 / 2175 / 2854 / 3288 for 0-2k / 2-4k / 4-6k /
  6-9k / 9k+; v5 in the arena 1329 / 1563 / 2840 / 4169 / 4437; v5 in the real games 2560 / 2017 / 2897 /
  4831 / 6888 (parked at the checkpoint while the runner was far); `blockLead=2500` in the arena 1348 /
  1502 / 2091 / 2106 / 2054. NB the arena cannot show the real failure: v4/v5 runners get pinned and stay
  near, so v5's blocker is within 2,500 of them 77-80% of turns in the arena vs 41-55% against top bots.
  Protocol for blocker changes: no regression in the arena AND a profile that matches the top bots.
  Six more v5 real games (2026-10-16; 909123192 MadKnight, 909123082 l4time, 909121718 magaiti, 909119183 pb4,
  909113931 Discourage — losses; 909123117 mykeich — win; all re-simulate exactly). Blocker engagement
  (share of turns within 2,500 of the enemy runner / median speed / hits on it), ours vs theirs:
  MadKnight 39%/312/11 vs 84%/202/50; pb4 49%/308/13 vs 72%/211/29; Discourage 49%/163/28 vs 78%/207/52;
  l4time 93%/248/70 vs 86%/190/60 (close game, lost on the final clock duel); magaiti 79%/258/51 vs
  75%/202/51 (close); mykeich 69%/309/17 vs 53%/291/19 (win). Engagement tracks the result: the games
  where our blocker stayed on their runner were the close ones. Our shields 23-55 per game vs their 7-24.
  Pinned-runner stats over these six (tools/pin_stats.py): ours thrust 0 on 8.8% of pinned turns, speed
  toward the checkpoint 20, escape speed 164, episode p90 40 turns; theirs 30% / 85 / 232 / p90 18.
- OUR LADDER GAMES (agent id 6796106; `python3 tools/fetch_replays.py --agent 6796106 --per-agent 100 --out
  data/ours`, then `python3 tools/our_games.py data/ours`). 2026-10-20: 100 games, all re-simulate exactly,
  39 wins / 61 losses, rank #36 of Legend (score 42.51; ranks 17-39 within 0.9 points). Losses vs wins:
  their blocker within 2,500 of our runner 85% vs 73% of turns, hits on it 57 vs 32 per game. Discourage
  (#13) beat us 19 of 21. Our runner while pinned (tools/pin_stats.py over data/ours.txt): in losses
  episodes median 9 / p90 71 turns, speed toward the checkpoint -22, thrust 0 on 7% of pinned turns;
  vs Discourage p90 78, speed -32 — e.g. game 909126973, one 253-turn pin: runner 1,000-1,900 from the
  checkpoint aiming within 30 deg of it at thrust 200, blocker 200-900 from the centre ramming it back.
  THE DECISIVE COMPARISON: the other top bots beat Discourage 55 of 60 (data/replays*), and their runners
  escape its blocker in a median 4 turns (p90 12) at speed +214 toward the checkpoint, coasting 28% and
  aiming >25 deg away 45% of pinned turns. Discourage's blocker is beatable; our runner is the problem.
- The arena DOES reproduce it: v5 vs v5, each runner is pinned ~208 turns per game, episodes p90 43-45,
  speed toward the checkpoint +4 (`python3 tools/arena_blocking.py DIR` now prints these runner-pin metrics).
  So runner fixes can be measured there: win rate AND these numbers moving toward the top bots'.
- Not a sparring partner: the imitation network alone (MPR_MODE=nn) — its blocker sits still (median speed 7),
  lost 0-40 to v5. A blockerless v5 (blockDistW=0,blockAngW=0) loses 0-200 to anything: uninformative.
- THE unsolved problem: our runner escaping a pin. Top bots' runners get past our blocker in 36-70 turns,
  ours stays 50-100 against theirs; mutual pins are stalemates decided by the clock. Escort and angle relief
  failed; untried: shield-and-push timing, approaching from the side at speed, a longer horizon only for
  the pinned runner, or a learned response from the top bots' own escapes (tools/replay_trace.py shows them).
- Score our plans against several opponent predictions (network + evolved) for robustness.
- Value net from game outcomes; MCTS over the discrete action set.
- Test `shieldUseW=600` on CodinGame itself: the arena opponent (v3) rams more than the top bots do.
