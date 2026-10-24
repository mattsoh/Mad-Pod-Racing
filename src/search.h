// Evolutionary search over 6-turn plans for both pods.
//
// Each turn: (1) evolve a predicted plan for the opponent against our previous
// plan, (2) evolve our plan against that prediction. Plans are kept between
// turns, shifted by one, so the search keeps improving rather than restarting.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>

#include "engine.h"
#include "rng.h"

namespace mpr {

#ifndef MPR_DEPTH
#define MPR_DEPTH 6
#endif
#ifndef MPR_POP
#define MPR_POP 6
#endif
constexpr int DEPTH = MPR_DEPTH;
constexpr int POP = MPR_POP;

struct Gene {
  float rot;       // degrees, -18..18
  int16_t thrust;  // 0..200
  uint8_t kind;    // K_THRUST, K_SHIELD, K_BOOST
};
struct Plan {
  Gene g[DEPTH][2];
};

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
};

// Evaluation weights. All terms are in map units (1 degree ~ angleW units).
struct EvalParams {
  double angleW = 4.0;        // runner: per degree off its next checkpoint
  double blockDistW = 1.2;    // blocker: per unit from its intercept point (0.6 let it dawdle: real losses show it 2-8k away)
  double blockAngW = 3.0;     // blocker: per degree off facing the enemy runner
  double timeoutW = 2500.0;   // per turn a team's timeout falls below 25
  double shieldW = 300.0;     // per thrustless shield turn still owed at the horizon
  double speedW = 0.0;        // runner: per unit of speed toward its next checkpoint
  double oppShare = 0.3;      // share of the turn spent evolving the opponent's plan (0 = trust the model)
  double rescueSpeed = 450;   // blocker's assumed average speed on the way to its own checkpoint
  double rescueTurn = 10;     // turns allowed for turning round and getting up to speed (15 was 55.5% vs 10 on top of robustW)
  double rescueSlack = 10;    // rescue ramps up over the last `rescueSlack` turns of slack and is full at zero slack
  double rescueW = 3.0;       // per unit of the blocker's distance to its own checkpoint, at full rescue
  double rescueAngW = 6.0;    // per degree the blocker is off heading for it
  double rescueSpeedW = 3.0;  // per unit of the blocker's speed toward it
  double rescueLead = 10;     // no rescue while our clock leads theirs by more than this (in a mutual pin, whoever breaks off first loses the race; 0 = off)
  double rescueOf[2] = {0, 0};  // set each turn from the root: how urgently each team's blocker must go (0..1)
  // Extra value of the BOOST still unused by the pod that is our blocker at the root. Top-15 bots fire 88% of
  // their boosts with the runner, 74% after turn 60 (to break pins and sprint); v5/v6 fired 55% with the
  // blocker and over half in the first 20 turns, then had none left for pins (0.1% of pinned turns vs 3.3%).
  double boostBlkW = 0;
  double boostEarly = 0;
  // Our runner's progress measured along the shortest path round the enemy pods (radius pathR) instead of in a
  // straight line: weight of the detour they force (0 = off). Whichever of our pods clears the line scores,
  // so the blocker knocks a pinning pod aside only when that frees the runner within the horizon.
  double pathW = 0, pathR = 800;
  // Cost per SHIELD by the pod that is our runner at the root. On the ladder v6's pinned runner spent 29% of
  // its pinned turns shielding or reloading (top-15 bots 14%), each shield costing 3 turns without thrust.
  double runShieldW = 500;     // v7. 0 = v6. vs v6 at 40 ms: 63.0% (126-74, seed 21); pinned shield/reload 24.5% -> 9.4%      // our unused boosts are worth boostW * (1 + boostEarly * share of the race still to run)
  int rootBlk[2] = {1, 3};    // set each turn from the root: each team's blocker
  double snapP = 0.125;       // chance a mutation snaps to the top bots' moves (-18/0/+18 deg, thrust 0/200)
  double boostW = 2500;       // value of each BOOST still unused at the horizon (top bots save theirs; 0 lost 39% vs v1, 2000-4000 won ~80%)
  double blockOff = 1000;     // blocker: waits this far in front of the checkpoint, toward the enemy runner (top bots: 700-1100)
  double blockLead = 1e9;     // ... but never more than this far ahead of the runner on its line to the checkpoint: top bots shadow the runner ~2,500 ahead instead of commuting to the checkpoint (1e9 = v5 behaviour)
  double blockStick = 3000;   // blocker: keeps defending the runner's next checkpoint unless the runner is this much closer to it (1500 in v4; 3000 = 57.5% vs it)
  double shieldUseW = 0;      // flat cost per SHIELD in our plan (top bots shield 1-7 times per 100 pod-turns; we did 6-10)
  double robustW = 1;         // weight of a plan's downside against a second (heuristic) opponent prediction (0 = off, 1 = worst case); doubles the cost of an evaluation. 1: 76% vs v4; 0.5: 71%; 2: 60%
  double reactShare = 0.3;    // share of the turn spent re-evolving our plan against the network's answer to it (0 = off)
  double pinNet = 0;          // pinned runner (an enemy between it and its checkpoint): play the imitation network's moves.
                              // Top bots escape Discourage's blocker in a median 4 turns (p90 12) at speed +214 toward the
                              // checkpoint; our search pushes into it (p90 78 turns, speed -32). 0 = off.
  double pinRobustW = -1;     // robustW to use while our runner is pinned (-1 = robustW): the heuristic second opinion
                              // predicts a perfect interceptor, so worst-of-two vetoes every way round the blocker.

  // "angleW=4,shieldW=300" style overrides (MPR_PARAMS), for local tuning.
  void parse(const char* spec) {
    std::string str = spec;
    size_t pos = 0;
    while (pos < str.size()) {
      size_t comma = str.find(',', pos);
      std::string kv = str.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      size_t eq = kv.find('=');
      if (eq != std::string::npos) {
        std::string k = kv.substr(0, eq);
        double v = std::atof(kv.c_str() + eq + 1);
        if (k == "angleW") angleW = v;
        else if (k == "blockDistW") blockDistW = v;
        else if (k == "blockAngW") blockAngW = v;
        else if (k == "timeoutW") timeoutW = v;
        else if (k == "shieldW") shieldW = v;
        else if (k == "speedW") speedW = v;
        else if (k == "oppShare") oppShare = v;
        else if (k == "rescueSpeed") rescueSpeed = v;
        else if (k == "rescueTurn") rescueTurn = v;
        else if (k == "rescueSlack") rescueSlack = v;
        else if (k == "rescueW") rescueW = v;
        else if (k == "rescueAngW") rescueAngW = v;
        else if (k == "rescueSpeedW") rescueSpeedW = v;
        else if (k == "rescueLead") rescueLead = v;
        else if (k == "snapP") snapP = v;
        else if (k == "boostW") boostW = v;
        else if (k == "blockOff") blockOff = v;
        else if (k == "blockLead") blockLead = v;
        else if (k == "boostBlkW") boostBlkW = v;
        else if (k == "boostEarly") boostEarly = v;
        else if (k == "pathW") pathW = v;
        else if (k == "pathR") pathR = v;
        else if (k == "runShieldW") runShieldW = v;
        else if (k == "blockStick") blockStick = v;
        else if (k == "shieldUseW") shieldUseW = v;
        else if (k == "robustW") robustW = v;
        else if (k == "reactShare") reactShare = v;
        else if (k == "pinNet") pinNet = v;
        else if (k == "pinRobustW") pinRobustW = v;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
};

inline double dist(double ax, double ay, double bx, double by) { return std::hypot(ax - bx, ay - by); }

// Extra length of the shortest path from (rx, ry) to (cx, cy) that goes round a disc of radius r at (bx, by),
// over the straight line (0 when the line is clear or the disc covers the target).
inline double detour(double rx, double ry, double cx, double cy, double bx, double by, double r) {
  double dR = dist(rx, ry, bx, by), dC = dist(cx, cy, bx, by);
  if (dC <= r) return 0;
  dR = std::max(dR, r + 1e-3);
  double cosT = ((rx - bx) * (cx - bx) + (ry - by) * (cy - by)) / (dR * dC);
  double arc = std::acos(std::max(-1.0, std::min(1.0, cosT))) - std::acos(r / dR) - std::acos(r / dC);
  if (arc <= 0) return 0;
  return std::sqrt(dR * dR - r * r) + std::sqrt(dC * dC - r * r) + r * arc - dist(rx, ry, cx, cy);
}

inline void applyGene(State& s, int i, const Gene& g) { applyMove(s, i, g.rot, g.thrust, g.kind); }

// Score of a state from `team`'s point of view; `depth` = turns simulated.
inline double evaluate(const State& s, const Track& tr, int team, int depth, const EvalParams& P) {
  if (s.winner >= 0) {
    double v = 1e9 - depth * 1e7;  // a win sooner beats a win later
    return s.winner == team ? v : -v;
  }
  int a = 2 * team, b = a + 1, c = 2 - 2 * team, d = c + 1;
  double pa = progress(s.p[a], tr), pb = progress(s.p[b], tr);
  double pc = progress(s.p[c], tr), pd = progress(s.p[d], tr);
  int run = pa >= pb ? a : b, blk = run == a ? b : a;
  int orun = pc >= pd ? c : d;
  // While the blocker is rescuing the team clock, the enemy runner's progress is
  // not its concern (weighted by a per-turn constant, so plans stay comparable).
  double rescue = P.rescueOf[team];
  double score = std::max(pa, pb) - (1 - rescue) * std::max(pc, pd);

  const Pod& R = s.p[run];
  double toCp = std::atan2(tr.y[R.next] - R.y, tr.x[R.next] - R.x) / DEG;
  score -= P.angleW * std::fabs(angleDiff(R.angle, toCp));
  if (P.speedW != 0) {
    double c = std::cos(toCp * DEG), sn = std::sin(toCp * DEG);
    score += P.speedW * (R.vx * c + R.vy * sn);
  }

  // Team clock: if neither pod passes a checkpoint for 100 turns the team loses.
  // When the clock gets close to the turns the blocker needs to reach its own
  // checkpoint (see Search::think), it stops blocking and goes, so a pinned
  // runner cannot lose the game on time. Heading and speed toward it are
  // rewarded too: a blocker facing away needs 10 turns just to turn round, so
  // within the horizon its distance alone barely moves and a distance-only pull
  // lost to keeping the enemy runner pinned (3 clock battles in 4 lost).
  const Pod& B = s.p[blk];
  if (rescue > 0) {
    double dB = dist(B.x, B.y, tr.x[B.next], tr.y[B.next]);
    double toB = std::atan2(tr.y[B.next] - B.y, tr.x[B.next] - B.x);
    score -= rescue * (P.rescueW * dB + P.rescueAngW * std::fabs(angleDiff(B.angle, toB / DEG)));
    score += rescue * P.rescueSpeedW * (B.vx * std::cos(toB) + B.vy * std::sin(toB));
  }

  // Blocker: wait at the enemy runner's next checkpoint if it can get there
  // first, otherwise at the one after; face the runner. The top bots' blockers
  // stand blockOff units in front of the checkpoint, on the runner's side, not
  // on its centre: a runner passing at the edge of the 600 radius slips by a
  // blocker on the centre.
  const Pod& O = s.p[orun];
  int n1 = O.next, n2 = n1 + 1 == tr.n ? 0 : n1 + 1;
  // Without the blockStick margin, the moment the runner edged closer than the
  // blocker (e.g. after a ram overshot), the blocker left for the next checkpoint
  // and the runner walked in: that is how striker_- and Peterpanda got free.
  bool first = dist(B.x, B.y, tr.x[n1], tr.y[n1]) < dist(O.x, O.y, tr.x[n1], tr.y[n1]) + P.blockStick;
  int tgt = first ? n1 : n2;
  double fx = first ? O.x : tr.x[n1], fy = first ? O.y : tr.y[n1];
  double ux = fx - tr.x[tgt], uy = fy - tr.y[tgt], ul = std::max(1.0, std::hypot(ux, uy));
  // Standing point: between the runner and its checkpoint, blockLead ahead of the runner at most,
  // blockOff in front of the checkpoint at least (and never on top of the runner).
  double off = std::min(P.blockOff, 0.5 * ul);
  if (first && P.blockLead < 1e8) off = std::min(std::max(off, ul - P.blockLead), ul - 400);
  double blocking = 1 - rescue;
  score -= blocking * P.blockDistW * dist(B.x, B.y, tr.x[tgt] + off * ux / ul, tr.y[tgt] + off * uy / ul);
  double toRun = std::atan2(O.y - B.y, O.x - B.x) / DEG;
  score -= blocking * P.blockAngW * std::fabs(angleDiff(B.angle, toRun));

  score -= P.shieldW * (R.shield + 0.3 * B.shield);
  if (P.pathW != 0) {
    double cx = tr.x[R.next], cy = tr.y[R.next], worst = 0;
    for (int e : {c, d}) worst = std::max(worst, detour(R.x, R.y, cx, cy, s.p[e].x, s.p[e].y, P.pathR));
    score -= P.pathW * worst;
  }
  if (P.boostW != 0) {
    if (g_sharedBoost) {
      score += P.boostW * ((s.teamBoostUsed[team] ? 0 : 1) - (s.teamBoostUsed[1 - team] ? 0 : 1));
    } else {
      double left = 1.0 - std::max(s.p[a].passed, s.p[b].passed) / (double)tr.total();
      score += P.boostW * (1 + P.boostEarly * left) * ((s.p[a].boostUsed ? 0 : 1) + (s.p[b].boostUsed ? 0 : 1));
      score -= P.boostW * ((s.p[c].boostUsed ? 0 : 1) + (s.p[d].boostUsed ? 0 : 1));
    }
  }
  if (P.boostBlkW != 0 && !s.p[P.rootBlk[team]].boostUsed) score += P.boostBlkW;
  score -= P.timeoutW * std::max(0, 25 - s.timeout[team]);
  score += P.timeoutW * std::max(0, 25 - s.timeout[1 - team]);
  return score;
}

// ---- Hand-written policies, used to seed the search ------------------------

inline Gene aimGene(const Pod& p, double tx, double ty, bool brake) {
  double want = std::atan2(ty - p.y, tx - p.x) / DEG;
  double d = angleDiff(p.angle, want);
  double rot = std::max(-MAX_ROT, std::min(MAX_ROT, d));
  double after = std::fabs(d - rot);
  int thrust = MAX_THRUST;
  if (brake) thrust = after >= 90 ? 0 : (int)(MAX_THRUST * std::min(1.0, (90 - after) / 60));
  return {(float)rot, (int16_t)thrust, K_THRUST};
}

inline Gene runnerGene(const State& s, const Track& tr, int i) {
  const Pod& p = s.p[i];
  return aimGene(p, tr.x[p.next] - 3 * p.vx, tr.y[p.next] - 3 * p.vy, true);
}

inline Gene blockerGene(const State& s, const Track& tr, int i, int target) {
  const Pod& p = s.p[i];
  const Pod& o = s.p[target];
  if (dist(p.x, p.y, o.x, o.y) < 3000) return aimGene(p, o.x + 2 * o.vx, o.y + 2 * o.vy, false);
  int n1 = o.next, n2 = n1 + 1 == tr.n ? 0 : n1 + 1;
  int tgt = dist(p.x, p.y, tr.x[n1], tr.y[n1]) < dist(o.x, o.y, tr.x[n1], tr.y[n1]) ? n1 : n2;
  return aimGene(p, tr.x[tgt] - 2 * p.vx, tr.y[tgt] - 2 * p.vy, true);
}

inline void roles(const State& s, const Track& tr, int team, int& run, int& blk) {
  int a = 2 * team;
  run = progress(s.p[a], tr) >= progress(s.p[a + 1], tr) ? a : a + 1;
  blk = run == a ? a + 1 : a;
}

// Rolls the heuristic forward for both teams and records `team`'s genes.
inline Plan heuristicPlan(const State& root, const Track& tr, int team) {
  Plan plan;
  State s = root;
  for (int d = 0; d < DEPTH; ++d) {
    Gene g[4];
    for (int t = 0; t < 2; ++t) {
      int run, blk, orun, oblk;
      roles(s, tr, t, run, blk);
      roles(s, tr, 1 - t, orun, oblk);
      g[run] = runnerGene(s, tr, run);
      g[blk] = blockerGene(s, tr, blk, orun);
    }
    plan.g[d][0] = g[2 * team];
    plan.g[d][1] = g[2 * team + 1];
    for (int k = 0; k < 4; ++k) applyGene(s, k, g[k]);
    step(s, tr);
    if (s.winner >= 0) {
      for (int e = d + 1; e < DEPTH; ++e) plan.g[e][0] = plan.g[d][0], plan.g[e][1] = plan.g[d][1];
      break;
    }
  }
  return plan;
}

// ---- Search -----------------------------------------------------------------

struct Search {
  Track tr;
  EvalParams P;
  Rng rng{12345};
  Plan mine{}, theirs{};
  bool haveMine = false, haveTheirs = false;
  const Plan* alt = nullptr;   // second opponent prediction while evolving our plan (robustW > 0)

  // A candidate's score against the predicted opponent plan, less robustW times
  // how much worse it does against the alternative prediction(s).
  double scoreRobust(const State& root, const Plan& cand, int team, const Plan& other) {
    double s1 = team == 0 ? score(root, cand, other, 0) : score(root, other, cand, 1);
    if (!alt) return s1;
    double s2 = team == 0 ? score(root, cand, *alt, 0) : score(root, *alt, cand, 1);
    return s1 + robW * std::min(0.0, s2 - s1);
  }
  double robW = 0;  // robustW in force this turn

  // Our runner is pinned: an enemy pod sits closer to its checkpoint than it does, within 2,500 of it.
  bool runnerPinned(const State& root, int& run) const {
    int blk;
    roles(root, tr, 0, run, blk);
    const Pod& R = root.p[run];
    double cx = tr.x[R.next], cy = tr.y[R.next], dR = dist(R.x, R.y, cx, cy);
    bool pinned = false;
    for (int e = 2; e < 4; ++e)
      pinned |= dR < 4000 && dist(root.p[e].x, root.p[e].y, cx, cy) < dR &&
                dist(root.p[e].x, root.p[e].y, R.x, R.y) < 2500;
    return pinned;
  }
  long evals = 0;
  double lastScore = 0;
  // Optional model of play (the network): fills `out` with `team`'s plan, rolled
  // forward while the other team follows `other`. Used to predict the opponent
  // and to seed our own search.
  std::function<void(const State&, int, const Plan&, Plan&)> model;

  double score(const State& root, const Plan& t0, const Plan& t1, int team) {
    ++evals;
    State s = root;
    const Plan& own = team == 0 ? t0 : t1;
    double cost = 0;
    for (int d = 0; d < DEPTH; ++d) {
      cost += P.shieldUseW * ((own.g[d][0].kind == K_SHIELD) + (own.g[d][1].kind == K_SHIELD));
      cost += P.runShieldW * (own.g[d][(P.rootBlk[team] & 1) ^ 1].kind == K_SHIELD);
      applyGene(s, 0, t0.g[d][0]);
      applyGene(s, 1, t0.g[d][1]);
      applyGene(s, 2, t1.g[d][0]);
      applyGene(s, 3, t1.g[d][1]);
      step(s, tr);
      if (s.winner >= 0) return evaluate(s, tr, team, d + 1, P) - cost;
    }
    return evaluate(s, tr, team, DEPTH, P) - cost;
  }

  Gene randomGene(bool boostOk) {
    Gene g;
    g.rot = (float)rng.range(-MAX_ROT, MAX_ROT);
    int r = rng.below(100);
    if (r < 3) return {g.rot, 0, K_SHIELD};
    if (boostOk && r < 4) return {g.rot, MAX_THRUST, K_BOOST};
    g.thrust = (int16_t)(r < 50 ? MAX_THRUST : rng.below(MAX_THRUST + 1));
    g.kind = K_THRUST;
    return g;
  }

  // Changes one gene; `amp` in (0,1] shrinks the step as time runs out.
  void mutate(Plan& p, double amp, const bool boostOk[2]) {
    int d = rng.below(DEPTH), k = rng.below(2);
    Gene& g = p.g[d][k];
    int r = rng.below(100);
    if (r < 4) {
      g.kind = g.kind == K_SHIELD ? K_THRUST : K_SHIELD;
      if (g.kind == K_THRUST) g.thrust = MAX_THRUST;
      return;
    }
    if (boostOk[k] && r < 6) {
      g.kind = g.kind == K_BOOST ? K_THRUST : K_BOOST;
      g.thrust = MAX_THRUST;
      return;
    }
    if (g.kind != K_THRUST) g.kind = K_THRUST, g.thrust = MAX_THRUST;
    if (r < 55) {
      if (rng.uniform() < P.snapP) g.rot = (float)((rng.below(3) - 1) * MAX_ROT);
      else g.rot = (float)std::max(-MAX_ROT, std::min(MAX_ROT, g.rot + amp * rng.range(-36, 36)));
    } else {
      if (rng.uniform() < P.snapP) g.thrust = (int16_t)(rng.below(2) * MAX_THRUST);
      else g.thrust = (int16_t)std::max(0.0, std::min((double)MAX_THRUST, g.thrust + amp * rng.range(-200, 200)));
    }
  }

  static Plan shifted(const Plan& p) {
    Plan q;
    for (int d = 0; d + 1 < DEPTH; ++d) q.g[d][0] = p.g[d + 1][0], q.g[d][1] = p.g[d + 1][1];
    q.g[DEPTH - 1][0] = p.g[DEPTH - 1][0];
    q.g[DEPTH - 1][1] = p.g[DEPTH - 1][1];
    for (int k = 0; k < 2; ++k)
      if (q.g[DEPTH - 1][k].kind != K_THRUST) q.g[DEPTH - 1][k] = {0, MAX_THRUST, K_THRUST};
    return q;
  }

  // Evolves `team`'s plan against a fixed plan for the other team.
  Plan evolve(const State& root, int team, const Plan& other, const Plan* seeds, int nSeeds, const Timer& clk,
              double tStart, double tEnd) {
    bool boostOk[2] = {boostAvailable(root, 2 * team), boostAvailable(root, 2 * team + 1)};
    Plan pop[POP];
    double sc[POP];
    for (int i = 0; i < POP; ++i) {
      if (i < nSeeds) pop[i] = seeds[i];
      else
        for (int d = 0; d < DEPTH; ++d)
          for (int k = 0; k < 2; ++k) pop[i].g[d][k] = randomGene(boostOk[k]);
      sc[i] = scoreRobust(root, pop[i], team, other);
    }
    int worst = 0;
    for (int i = 1; i < POP; ++i)
      if (sc[i] < sc[worst]) worst = i;
    double amp = 1.0;
    for (int it = 0;; ++it) {
      if ((it & 7) == 0) {
        double now = clk.ms();
        if (now >= tEnd) break;
        amp = std::max(0.1, 1.0 - (now - tStart) / std::max(1.0, tEnd - tStart));
      }
      int a = rng.below(POP), b = rng.below(POP);
      Plan child = pop[sc[a] >= sc[b] ? a : b];
      mutate(child, amp, boostOk);
      if (amp > 0.6 && rng.below(2)) mutate(child, amp, boostOk);
      double s = scoreRobust(root, child, team, other);
      if (s > sc[worst]) {
        pop[worst] = child;
        sc[worst] = s;
        for (int i = 0; i < POP; ++i)
          if (sc[i] < sc[worst]) worst = i;
      }
    }
    int best = 0;
    for (int i = 1; i < POP; ++i)
      if (sc[i] > sc[best]) best = i;
    lastScore = sc[best];
    return pop[best];
  }

  // Our team is 0 (pods 0 and 1). Returns the plan whose first move to play.
  const Plan& think(const State& root, double budgetMs) {
    double oppShare = P.oppShare;
    Timer clk;
    evals = 0;
    // How urgently each team's blocker must go for its own checkpoint: full once
    // the clock is within rescueSlack turns of the time it needs to get there.
    for (int t = 0; t < 2; ++t) {
      int run, blk;
      roles(root, tr, t, run, blk);
      P.rootBlk[t] = blk;
      const Pod& B = root.p[blk];
      double need = dist(B.x, B.y, tr.x[B.next], tr.y[B.next]) / P.rescueSpeed + P.rescueTurn;
      P.rescueOf[t] = std::max(0.0, std::min(1.0, (P.rescueSlack - (root.timeout[t] - need)) / P.rescueSlack));
      if (P.rescueLead > 0 && root.timeout[t] > root.timeout[1 - t] + P.rescueLead) P.rescueOf[t] = 0;
    }
    Plan h0 = heuristicPlan(root, tr, 0), h1 = heuristicPlan(root, tr, 1);
    Plan myPrev = haveMine ? shifted(mine) : h0;
    Plan opPrev = haveTheirs ? shifted(theirs) : h1;
    Plan oppSeeds[3] = {opPrev, h1}, mySeeds[5] = {myPrev, h0};
    int nSeeds = 2;
    if (model) {
      model(root, 1, myPrev, oppSeeds[2]);  // the opponent, reacting to what we plan to do
      nSeeds = 3;
    }
    if (oppShare > 0 || !model)
      theirs = evolve(root, 1, myPrev, oppSeeds, nSeeds, clk, clk.ms(), budgetMs * oppShare);
    else
      theirs = oppSeeds[2];  // trust the model's prediction as is
    if (model) model(root, 0, theirs, mySeeds[2]);
    int myN = nSeeds;
    int run;
    bool pinned = runnerPinned(root, run);
    robW = pinned && P.pinRobustW >= 0 ? P.pinRobustW : P.robustW;
    alt = robW > 0 && model ? &h1 : nullptr;  // the heuristic opponent as the second opinion
    bool react = model && P.reactShare > 0;
    mine = evolve(root, 0, theirs, mySeeds, myN, clk, clk.ms(), react ? budgetMs * (1 - P.reactShare) : budgetMs);
    if (react) {  // predict their answer to the plan we found, then improve ours against that
      model(root, 1, mine, theirs);
      Plan seeds2[2] = {mine, mine};
      model(root, 0, theirs, seeds2[1]);
      mine = evolve(root, 0, theirs, seeds2, 2, clk, clk.ms(), budgetMs);
    }
    if (pinned && P.pinNet > 0 && model) {  // pinned runner: move like the top bots' runners, which the network imitates
      Plan net;
      model(root, 0, theirs, net);
      for (int d = 0; d < DEPTH; ++d) mine.g[d][run] = net.g[d][run];
    }
    alt = nullptr;
    haveMine = haveTheirs = true;
    return mine;
  }
};

}  // namespace mpr
