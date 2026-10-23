// Mad Pod Racing bot (Legend rules) - search bot with an optional neural net.
//
// Build locally:   make bot          Submit:  make bundle  ->  dist/mad_pod_bot.cpp
#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")

// What runs on CodinGame (there is no environment there):
//   "auto"      hybrid if a network is embedded, otherwise search
//   "search"    evolutionary search only
//   "hybrid"    search, seeded each turn with the network's plans for both teams
//   "nn"        the network alone (two duels per turn, no search)
//   "heuristic" hand-written baseline
// Locally, MPR_MODE overrides it.
#ifndef MPR_DEFAULT_MODE
#define MPR_DEFAULT_MODE "auto"
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "engine.h"
#include "nnbot.h"
#include "search.h"

using namespace mpr;

namespace {

struct Tracker {
  int laps = 3, n = 0, turn = 0;
  int prevNext[4] = {}, passed[4] = {};
  int timeout[2] = {TIMEOUT_TURNS, TIMEOUT_TURNS};
  int myShield[2] = {};                 // shield value at the start of the turn
  bool podBoost[4] = {}, teamBoost[2] = {};
  State prev{};
};

// Pods can only have collided if they were close enough at the start of the turn.
bool couldCollide(const State& s, int i) {
  for (int j = 0; j < 4; ++j) {
    if (j == i) continue;
    double reach = 800 + std::hypot(s.p[i].vx, s.p[i].vy) + std::hypot(s.p[j].vx, s.p[j].vy) + 2 * BOOST_THRUST;
    if (dist(s.p[i].x, s.p[i].y, s.p[j].x, s.p[j].y) < reach) return true;
  }
  return false;
}

// Prediction of one of our pods, made when no collision is possible, to check
// the engine against the real game every turn.
struct Prediction {
  bool valid = false;
  int k = 0;
  Pod p{};
};

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  // 55 ms against the 75 ms limit: at 60 ms the worst of 3,694 local turns took 67.7 ms,
  // and a single late answer forfeits the game.
  double budget = 55, firstBudget = 800;
  if (const char* e = std::getenv("MPR_TIME_MS")) {
    budget = std::atof(e);
    firstBudget = std::min(800.0, budget * 10);
  }
  const char* modeEnv = std::getenv("MPR_MODE");
  std::string mode = modeEnv ? modeEnv : MPR_DEFAULT_MODE;
  if (mode == "auto") mode = nnAvailable() ? "hybrid" : "search";
  if ((mode == "nn" || mode == "hybrid") && !nnAvailable()) mode = "search";

  std::cerr << "mode " << mode << (nnAvailable() ? " (network embedded)" : " (no network)") << ", " << budget
            << " ms per turn\n";

  Tracker tk;
  Track tr;
  std::cin >> tk.laps >> tk.n;
  tr.laps = tk.laps;
  tr.n = tk.n;
  for (int i = 0; i < tr.n; ++i) std::cin >> tr.x[i] >> tr.y[i];

  Search S;
  S.tr = tr;
  S.rng = Rng(1234567 + (uint64_t)tr.x[0] * 31 + (uint64_t)tr.y[1]);
  // With the imitation network predicting the opponent (twice as accurately as
  // evolving its plan, measured on real top-bot games), trust it and spend the
  // whole turn on our own plan.
  if (mode == "hybrid") S.P.oppShare = bcNet() ? 0.0 : 0.1;
#ifdef MPR_PARAMS_BAKED  // variant builds for CodinGame test games, which have no environment
  S.P.parse(MPR_PARAMS_BAKED);
#endif
  if (const char* e = std::getenv("MPR_PARAMS")) S.P.parse(e);
  if (std::getenv("MPR_SHARED_BOOST")) g_sharedBoost = true;  // local experiments only
  NNPlayer nn(tr);
  if (mode == "hybrid")
    S.model = [&nn](const State& s, int team, const Plan& other, Plan& p) { nn.rollout(s, team, other, p); };
  Prediction pred[2];
  int mismatches = 0;

  for (;; ++tk.turn) {
    int in[4][6];
    if (!(std::cin >> in[0][0])) break;
    Timer clk;
    for (int j = 1; j < 6; ++j) std::cin >> in[0][j];
    for (int k = 1; k < 4; ++k)
      for (int j = 0; j < 6; ++j) std::cin >> in[k][j];

    // --- Rebuild the full state, including what the input leaves out --------
    State st{};
    bool teamPassed[2] = {false, false};
    for (int k = 0; k < 4; ++k) {
      if (tk.turn > 0) {
        int adv = (in[k][5] - tk.prevNext[k] + tr.n) % tr.n;
        tk.passed[k] += adv;
        if (adv) teamPassed[k >> 1] = true;
      }
      tk.prevNext[k] = in[k][5];
    }
    for (int t = 0; t < 2; ++t)
      if (tk.turn > 0) tk.timeout[t] = teamPassed[t] ? TIMEOUT_TURNS - 1 : tk.timeout[t] - 1;

    // Opponent boost: an acceleration beyond max thrust with no pod nearby.
    if (tk.turn > 0)
      for (int k = 2; k < 4; ++k) {
        if (couldCollide(tk.prev, k)) continue;
        double ax = in[k][2] / FRICTION - tk.prev.p[k].vx, ay = in[k][3] / FRICTION - tk.prev.p[k].vy;
        if (std::hypot(ax, ay) > MAX_THRUST + 60) tk.podBoost[k] = tk.teamBoost[1] = true;
      }

    for (int k = 0; k < 4; ++k) {
      Pod& p = st.p[k];
      p.x = in[k][0], p.y = in[k][1], p.vx = in[k][2], p.vy = in[k][3];
      p.next = in[k][5];
      p.passed = tk.passed[k];
      p.angle = tk.turn == 0 ? normAngle(std::atan2(tr.y[p.next] - p.y, tr.x[p.next] - p.x) / DEG)
                             : normAngle(in[k][4]);
      p.shield = k < 2 ? tk.myShield[k] : 0;
      p.boostUsed = tk.podBoost[k];
      if (tk.turn > 1) observeLastMove(p, tk.prev.p[k]);
      else clearLastMove(p);
    }
    st.timeout[0] = tk.timeout[0];
    st.timeout[1] = tk.timeout[1];
    st.teamBoostUsed[0] = tk.teamBoost[0];
    st.teamBoostUsed[1] = tk.teamBoost[1];
    st.turn = tk.turn;
    st.winner = -1;

    // --- Self-check: did our pods end up where the engine said? ---------------
    for (auto& pr : pred) {
      if (!pr.valid) continue;
      const Pod& a = pr.p;
      const Pod& b = st.p[pr.k];
      int ang = ((int)roundHalfUp(a.angle)) % 360;
      if (a.x != b.x || a.y != b.y || a.vx != b.vx || a.vy != b.vy || ang != in[pr.k][4] || a.next != b.next) {
        if (mismatches++ < 5)
          std::cerr << "SIMCHECK turn " << tk.turn << " pod " << pr.k << " predicted " << a.x << ' ' << a.y << ' '
                    << a.vx << ' ' << a.vy << ' ' << ang << ' ' << a.next << " got " << b.x << ' ' << b.y << ' '
                    << b.vx << ' ' << b.vy << ' ' << in[pr.k][4] << ' ' << b.next << '\n';
      }
      pr.valid = false;
    }

    // --- Decide ---------------------------------------------------------------
    double limit = tk.turn == 0 ? firstBudget : budget;
    Gene out[2];
    if (mode == "nn") {
      nn.act(st, nullptr, out);
    } else if (mode == "heuristic") {
      Plan h = heuristicPlan(st, tr, 0);
      out[0] = h.g[0][0];
      out[1] = h.g[0][1];
      if (tk.turn == 0) out[0].kind = K_BOOST;
    } else {
      const Plan& plan = S.think(st, limit);
      out[0] = plan.g[0][0];
      out[1] = plan.g[0][1];
    }

    // --- Output ---------------------------------------------------------------
    for (int k = 0; k < 2; ++k) {
      const Pod& p = st.p[k];
      double a = (p.angle + out[k].rot) * DEG;
      long tx = (long)roundHalfUp(p.x + 10000 * std::cos(a));
      long ty = (long)roundHalfUp(p.y + 10000 * std::sin(a));
      std::cout << tx << ' ' << ty << ' ';
      if (out[k].kind == K_SHIELD) std::cout << "SHIELD";
      else if (out[k].kind == K_BOOST && boostAvailable(st, k)) std::cout << "BOOST";
      else std::cout << out[k].thrust;
      if (k == 0) std::cout << ' ' << (mode == "search" || mode == "hybrid" ? std::to_string(S.evals / 1000) + "k" : mode);
      std::cout << '\n';

      // Predict this pod alone; only checked if nothing could hit it.
      pred[k].valid = tk.turn > 0 && !couldCollide(st, k) && std::fabs(out[k].rot) < 17.5;
      if (pred[k].valid) {
        State iso = st;
        for (int j = 0; j < 4; ++j)
          if (j != k) iso.p[j].x = -1e6 * (j + 1), iso.p[j].vx = iso.p[j].vy = 0;
        applyCommand(iso, k, (double)tx, (double)ty, out[k].thrust, out[k].kind, false);
        step(iso, tr);
        pred[k].k = k;
        pred[k].p = iso.p[k];
      }

      if (out[k].kind == K_SHIELD) tk.myShield[k] = SHIELD_TURNS - 1;
      else if (tk.myShield[k] > 0) tk.myShield[k]--;
      if (out[k].kind == K_BOOST && boostAvailable(st, k)) {
        tk.podBoost[k] = true;
        tk.teamBoost[0] = true;
        st.p[k].boostUsed = true;
        st.teamBoostUsed[0] = true;
      }
    }
    std::cout.flush();
    tk.prev = st;
    std::cerr << "turn " << tk.turn << " " << clk.ms() << "ms evals " << S.evals << " score " << (long)S.lastScore
              << '\n';
  }
  return 0;
}
