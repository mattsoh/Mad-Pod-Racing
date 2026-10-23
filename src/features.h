// Network inputs and actions for the runner-vs-blocker duel.
//
// Everything is expressed in the runner's frame: origin at the runner, +x
// pointing at its next checkpoint, so the same corner looks the same anywhere
// on the map.
#pragma once
#include <cmath>

#include "engine.h"
#include "search.h"

namespace mpr {

constexpr int N_IN = 22;
constexpr int N_ACT = 7;                 // per pod
constexpr int N_OUT = N_ACT * N_ACT;     // Q[runner action][blocker action]

// Actions 0-5: rotate -18 / 0 / +18 with thrust 200 (0-2) or 0 (3-5). 6: SHIELD.
inline Gene actionGene(int a) {
  if (a == 6) return {0.f, 0, K_SHIELD};
  return {(float)((a % 3 - 1) * MAX_ROT), (int16_t)(a < 3 ? MAX_THRUST : 0), K_THRUST};
}

// blk < 0 means no blocker (solo runner).
inline void duelFeatures(const State& s, const Track& tr, int run, int blk, float* f) {
  const Pod& R = s.p[run];
  int c1 = R.next, c2 = c1 + 1 == tr.n ? 0 : c1 + 1, c3 = c2 + 1 == tr.n ? 0 : c2 + 1;
  double dx = tr.x[c1] - R.x, dy = tr.y[c1] - R.y;
  double d = std::sqrt(dx * dx + dy * dy);
  double ux = 1, uy = 0;
  if (d > 1e-6) ux = dx / d, uy = dy / d;
  else ux = std::cos(R.angle * DEG), uy = std::sin(R.angle * DEG);
  auto rx = [&](double vx, double vy) { return (float)(vx * ux + vy * uy); };
  auto ry = [&](double vx, double vy) { return (float)(-vx * uy + vy * ux); };
  double hc = std::cos(R.angle * DEG), hs = std::sin(R.angle * DEG);

  f[0] = (float)(d / 10000);
  f[1] = rx(hc, hs);
  f[2] = ry(hc, hs);
  f[3] = rx(R.vx, R.vy) / 1000;
  f[4] = ry(R.vx, R.vy) / 1000;
  f[5] = rx(tr.x[c2] - tr.x[c1], tr.y[c2] - tr.y[c1]) / 10000;
  f[6] = ry(tr.x[c2] - tr.x[c1], tr.y[c2] - tr.y[c1]) / 10000;
  f[7] = rx(tr.x[c3] - tr.x[c2], tr.y[c3] - tr.y[c2]) / 10000;
  f[8] = ry(tr.x[c3] - tr.x[c2], tr.y[c3] - tr.y[c2]) / 10000;
  f[9] = R.shield / (float)SHIELD_TURNS;
  f[21] = s.timeout[run >> 1] / (float)TIMEOUT_TURNS;
  if (blk < 0) {
    for (int i = 10; i < 21; ++i) f[i] = 0;
    return;
  }
  const Pod& B = s.p[blk];
  double bc = std::cos(B.angle * DEG), bs = std::sin(B.angle * DEG);
  f[10] = 1;
  f[11] = rx(B.x - R.x, B.y - R.y) / 10000;
  f[12] = ry(B.x - R.x, B.y - R.y) / 10000;
  f[13] = rx(B.vx, B.vy) / 1000;
  f[14] = ry(B.vx, B.vy) / 1000;
  f[15] = rx(bc, bs);
  f[16] = ry(bc, bs);
  f[17] = B.shield / (float)SHIELD_TURNS;
  f[18] = (float)(std::hypot(B.x - R.x, B.y - R.y) / 10000);
  f[19] = rx(B.x - tr.x[c1], B.y - tr.y[c1]) / 10000;
  f[20] = ry(B.x - tr.x[c1], B.y - tr.y[c1]) / 10000;
}

// ---- Full-game features for imitating top bots ------------------------------
//
// Everything about the race from pod i's point of view: origin at the pod,
// +x toward its next checkpoint. The other pods are its partner, then the
// enemy runner and enemy blocker (whichever is further along is the runner).

constexpr int N_FULL = 48;
constexpr int F_LAST = 44;  // last-move inputs: rotation, thrust, touched, masked

// What we can tell about each pod's last move from two consecutive inputs: the rotation from the integer
// angles, and the thrust from the speed change, which without a touch points along the new heading (to
// within the truncation of the speeds). `prev` and `s` are the states at the start of the last turn and of
// this one. The bot, the trainer and the evaluator all go through this, so training sees what play sees.
inline void observeLastMove(Pod& p, const Pod& prev) {
  double c = std::cos(p.angle * DEG), sn = std::sin(p.angle * DEG);
  double ax = p.vx / FRICTION - prev.vx, ay = p.vy / FRICTION - prev.vy;
  double along = ax * c + ay * sn, side = -ax * sn + ay * c;
  p.lastRot = (float)angleDiff(prev.angle, p.angle);
  p.lastTouch = std::fabs(side) > 2.5 + 0.01 * std::fabs(along) || along < -3 ||
                (along > MAX_THRUST + 3 && std::fabs(along - BOOST_THRUST) > 3);
  p.lastThrust = p.lastTouch ? 0.f : (float)along;
  p.lastMasked = false;
}
inline void clearLastMove(Pod& p) { p.lastRot = p.lastThrust = 0, p.lastTouch = p.lastMasked = false; }

inline void fullFeatures(const State& s, const Track& tr, int i, float* f) {
  const Pod& P = s.p[i];
  int t = i >> 1, mate = i ^ 1, e0 = 2 - 2 * t, e1 = e0 + 1;
  int erun = progress(s.p[e0], tr) >= progress(s.p[e1], tr) ? e0 : e1, eblk = erun ^ 1;
  int c1 = P.next, c2 = (c1 + 1) % tr.n, c3 = (c2 + 1) % tr.n;
  double dx = tr.x[c1] - P.x, dy = tr.y[c1] - P.y, d = std::sqrt(dx * dx + dy * dy);
  double ux = 1, uy = 0;
  if (d > 1e-6) ux = dx / d, uy = dy / d;
  else ux = std::cos(P.angle * DEG), uy = std::sin(P.angle * DEG);
  auto rx = [&](double vx, double vy) { return (float)(vx * ux + vy * uy); };
  auto ry = [&](double vx, double vy) { return (float)(-vx * uy + vy * ux); };
  int k = 0;
  // self
  f[k++] = (float)(d / 10000);
  f[k++] = rx(std::cos(P.angle * DEG), std::sin(P.angle * DEG));
  f[k++] = ry(std::cos(P.angle * DEG), std::sin(P.angle * DEG));
  f[k++] = rx(P.vx, P.vy) / 1000;
  f[k++] = ry(P.vx, P.vy) / 1000;
  f[k++] = rx(tr.x[c2] - P.x, tr.y[c2] - P.y) / 10000;
  f[k++] = ry(tr.x[c2] - P.x, tr.y[c2] - P.y) / 10000;
  f[k++] = rx(tr.x[c3] - P.x, tr.y[c3] - P.y) / 10000;
  f[k++] = ry(tr.x[c3] - P.x, tr.y[c3] - P.y) / 10000;
  f[k++] = P.shield / (float)SHIELD_TURNS;
  // race state
  double pm = progress(s.p[mate], tr), pi = progress(P, tr);
  f[k++] = pi >= pm ? 1.f : 0.f;
  f[k++] = boostAvailable(s, i) ? 1.f : 0.f;
  f[k++] = (float)(tr.total() - P.passed) / 24.f;
  f[k++] = s.timeout[t] / (float)TIMEOUT_TURNS;
  // the three other pods
  for (int o : {mate, erun, eblk}) {
    const Pod& Q = s.p[o];
    f[k++] = rx(Q.x - P.x, Q.y - P.y) / 10000;
    f[k++] = ry(Q.x - P.x, Q.y - P.y) / 10000;
    f[k++] = rx(Q.vx, Q.vy) / 1000;
    f[k++] = ry(Q.vx, Q.vy) / 1000;
    f[k++] = rx(std::cos(Q.angle * DEG), std::sin(Q.angle * DEG));
    f[k++] = ry(std::cos(Q.angle * DEG), std::sin(Q.angle * DEG));
    f[k++] = rx(tr.x[Q.next] - P.x, tr.y[Q.next] - P.y) / 10000;
    f[k++] = ry(tr.x[Q.next] - P.x, tr.y[Q.next] - P.y) / 10000;
    f[k++] = Q.shield / (float)SHIELD_TURNS;
  }
  f[k++] = s.timeout[1 - t] / (float)TIMEOUT_TURNS;
  double lead = std::max(pi, pm) - progress(s.p[erun], tr);
  f[k++] = (float)std::max(-2.0, std::min(2.0, lead / 50000));
  f[k++] = boostAvailable(s, erun) ? 1.f : 0.f;
  // last move (F_LAST): known only at the root of a rollout
  bool known = !P.lastMasked;
  f[k++] = known ? P.lastRot / (float)MAX_ROT : 0.f;
  f[k++] = known && !P.lastTouch ? std::max(0.f, std::min((float)BOOST_THRUST, P.lastThrust)) / MAX_THRUST : 0.f;
  f[k++] = known && P.lastTouch ? 1.f : 0.f;
  f[k++] = known ? 0.f : 1.f;
}

// Imitation outputs. Top bots only ever rotate -18, 0 or +18 degrees and
// thrust 0 or 200 (plus SHIELD / BOOST), so that is the action set.
constexpr int BC_ROT = 3, BC_KIND = 4, BC_OUT = BC_ROT + BC_KIND;
inline float bcRotCenter(int b) { return (float)((b - 1) * MAX_ROT); }
inline int bcRotBin(double rot) { return rot < -MAX_ROT / 2 ? 0 : rot > MAX_ROT / 2 ? 2 : 1; }
// kinds: 0 = thrust 0, 1 = thrust 200, 2 = SHIELD, 3 = BOOST
inline int bcKind(int thrust, int kind) {
  if (kind == K_SHIELD) return 2;
  if (kind == K_BOOST) return 3;
  return thrust >= MAX_THRUST / 2 ? 1 : 0;
}
inline Gene bcGene(int rotBin, int kindBin) {
  float r = bcRotCenter(rotBin);
  if (kindBin == 2) return {r, 0, K_SHIELD};
  if (kindBin == 3) return {r, MAX_THRUST, K_BOOST};
  return {r, (int16_t)(kindBin ? MAX_THRUST : 0), K_THRUST};
}

}  // namespace mpr
