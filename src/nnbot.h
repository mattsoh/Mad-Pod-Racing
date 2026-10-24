// Plays the game with an embedded network. Two kinds are supported, told
// apart by their input/output sizes:
//
//  - imitation net (tools/bc.cpp): one pod's view of the whole race in, a
//    rotation (-18 / 0 / +18) and a thrust kind (0 / 200 / SHIELD / BOOST) out,
//    learned from the top Legend bots' moves;
//  - duel net (tools/train.cpp): Q-values for runner-vs-blocker action pairs,
//    solved as a matrix game each turn.
#pragma once
#include "features.h"
// @bundle-skip-begin
#include "nash.h"
// @bundle-skip-end
#include "nn.h"
#include "search.h"
#include "weights.h"

namespace mpr {

inline const MLP& botNet() {
  static MLP m;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    if (NN_WEIGHTS[0]) dequantize(z85Decode(NN_WEIGHTS), m);
  }
  return m;
}
// Networks from before the last-move inputs (44 inputs, e.g. bc160_t15) still work: they read the first 44.
inline bool bcNet() {
  return (botNet().inputs() == N_FULL || botNet().inputs() == F_LAST) && botNet().outputs() == BC_OUT;
}
// @bundle-skip-begin
inline bool duelNet() { return botNet().inputs() == N_IN && botNet().outputs() == N_OUT; }
inline bool nnAvailable() { return bcNet() || duelNet(); }
// @bundle-skip-end
// @bundle-only inline bool nnAvailable() { return bcNet(); }

struct NNPlayer {
  Track tr;
  explicit NNPlayer(const Track& t) : tr(t) {}

  static int pickFrom(const float* p, int n, Rng* rng) {
    int best = 0;
    for (int i = 1; i < n; ++i)
      if (p[i] > p[best]) best = i;
    if (!rng) return best;
    double u = rng->uniform(), c = 0;
    for (int i = 0; i < n; ++i)
      if (u < (c += p[i])) return i;
    return best;
  }

  static void softmax(const float* z, int n, float* p) {
    float m = z[0], sum = 0;
    for (int i = 1; i < n; ++i) m = std::max(m, z[i]);
    for (int i = 0; i < n; ++i) sum += p[i] = std::exp(z[i] - m);
    for (int i = 0; i < n; ++i) p[i] /= sum;
  }

  // Imitation policy for pod k. probs (optional) receives rotation then kind probabilities.
  Gene bcMove(const State& s, int k, Rng* rng, float* probs = nullptr) const {
    float f[N_FULL], q[BC_OUT], pr[BC_ROT], pk[BC_KIND];
    fullFeatures(s, tr, k, f);
    botNet().forward(f, q);
    softmax(q, BC_ROT, pr);
    softmax(q + BC_ROT, BC_KIND, pk);
    if (!boostAvailable(s, k)) {
      pk[1] += pk[3];  // BOOST unavailable means full thrust
      pk[3] = 0;
    }
    if (probs) {
      for (int i = 0; i < BC_ROT; ++i) probs[i] = pr[i];
      for (int i = 0; i < BC_KIND; ++i) probs[BC_ROT + i] = pk[i];
    }
    return bcGene(pickFrom(pr, BC_ROT, rng), pickFrom(pk, BC_KIND, rng));
  }

  // @bundle-skip-begin
  void duel(const State& s, int run, int blk, float* xr, float* yb) const {
    float f[N_IN], q[N_OUT];
    duelFeatures(s, tr, run, blk, f);
    botNet().forward(f, q);
    solveGame(q, N_ACT, N_ACT, xr, yb);
  }
  // @bundle-skip-end

  // Moves for all four pods. Greedy when rng is null, sampled otherwise.
  void policy(const State& s, Rng* rng, Gene g[4]) const {
    if (bcNet()) {
      for (int k = 0; k < 4; ++k) g[k] = bcMove(s, k, rng);
      return;
    }
    // @bundle-skip-begin
    int run[2], blk[2];
    roles(s, tr, 0, run[0], blk[0]);
    roles(s, tr, 1, run[1], blk[1]);
    for (int t = 0; t < 2; ++t) {
      float x[N_ACT], y[N_ACT];
      duel(s, run[t], blk[1 - t], x, y);
      g[run[t]] = actionGene(pickFrom(x, N_ACT, rng));
      g[blk[1 - t]] = actionGene(pickFrom(y, N_ACT, rng));
    }
    // BOOST is not one of the duel net's actions: fire it on a long, aligned straight.
    for (int t = 0; t < 2; ++t) {
      const Pod& p = s.p[run[t]];
      Gene& gr = g[run[t]];
      if (!boostAvailable(s, run[t]) || gr.kind != K_THRUST || gr.thrust < MAX_THRUST) continue;
      double toCp = std::atan2(tr.y[p.next] - p.y, tr.x[p.next] - p.x) / DEG;
      double d = dist(p.x, p.y, tr.x[p.next], tr.y[p.next]);
      if (std::fabs(angleDiff(p.angle + gr.rot, toCp)) < 4 && d > 5000) gr.kind = K_BOOST;
    }
    // @bundle-skip-end
  }

  void act(const State& s, Rng* rng, Gene out[2]) const {
    Gene g[4];
    policy(s, rng, g);
    out[0] = g[0];
    out[1] = g[1];
  }

  // `team` plays the network (greedy), reacting turn by turn, while the other
  // team follows `other`; records `team`'s moves.
  void rollout(const State& root, int team, const Plan& other, Plan& plan) const {
    State s = root;
    for (int d = 0; d < DEPTH; ++d) {
      Gene g[4];
      for (int j = 0; j < 2; ++j) {
        int k = 2 * team + j;
        g[k] = bcNet() ? bcMove(s, k, nullptr) : Gene{0, MAX_THRUST, K_THRUST};
        g[k ^ 2] = other.g[d][j];
      }
      // @bundle-skip-begin
      if (!bcNet()) {
        Gene all[4];
        policy(s, nullptr, all);
        g[2 * team] = all[2 * team];
        g[2 * team + 1] = all[2 * team + 1];
      }
      // @bundle-skip-end
      plan.g[d][0] = g[2 * team];
      plan.g[d][1] = g[2 * team + 1];
      for (int k = 0; k < 4; ++k) applyGene(s, k, g[k]);
      step(s, tr);
      if (s.winner >= 0) {
        for (int e = d + 1; e < DEPTH; ++e) plan.g[e][0] = plan.g[d][0], plan.g[e][1] = plan.g[d][1];
        break;
      }
    }
  }

  // The network's greedy plan for `team`, rolled forward with all pods on the net.
  void rollout(const State& root, int team, Plan& plan) const {
    State s = root;
    for (int d = 0; d < DEPTH; ++d) {
      Gene g[4];
      policy(s, nullptr, g);
      plan.g[d][0] = g[2 * team];
      plan.g[d][1] = g[2 * team + 1];
      for (int k = 0; k < 4; ++k) applyGene(s, k, g[k]);
      step(s, tr);
      if (s.winner >= 0) {
        for (int e = d + 1; e < DEPTH; ++e) plan.g[e][0] = plan.g[d][0], plan.g[e][1] = plan.g[d][1];
        break;
      }
    }
  }
};

}  // namespace mpr
