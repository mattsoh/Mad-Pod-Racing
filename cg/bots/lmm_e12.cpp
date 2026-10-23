#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")
//   "auto"      hybrid if a network is embedded, otherwise search
//   "search"    evolutionary search only
//   "hybrid"    search, seeded each turn with the network's plans for both teams
//   "nn"        the network alone (two duels per turn, no search)
//   "heuristic" hand-written baseline
#define MPR_LAST_MOVE 1  // the network's inputs include each pod's last move
#ifndef MPR_DEFAULT_MODE
#define MPR_DEFAULT_MODE "auto"
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
// Follows the Gold/Legend "Expert rules" and robostac's referee, which was
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
namespace mpr {
constexpr double PI_D = 3.14159265358979323846;
constexpr double DEG = PI_D / 180.0;
constexpr int MAX_THRUST = 200;       // Legend: 0..200
constexpr int BOOST_THRUST = 650;
constexpr double POD_RSQ = 640000.0;  // (400 + 400)^2
constexpr double CP_RADIUS = 600.0;
constexpr double CP_RSQ = 360000.0;
constexpr double FRICTION = 0.85;
constexpr double MAX_ROT = 18.0;
constexpr int TIMEOUT_TURNS = 100;
constexpr int SHIELD_TURNS = 4;       // activation turn + 3 turns without thrust
constexpr int MAX_CP = 8;
inline bool g_sharedBoost = false;
enum Kind : uint8_t { K_THRUST = 0, K_SHIELD = 1, K_BOOST = 2 };
struct Track {
  int n = 0, laps = 3;
  double x[MAX_CP] = {}, y[MAX_CP] = {};
  int total() const { return n * laps; }
};
struct Pod {
  double x, y, vx, vy;
  double angle;    // degrees in [0, 360)
  double rad;      // the same heading kept in radians the way the referee keeps it
  int next;        // index of the next checkpoint
  int passed;      // checkpoints passed so far
  int shield;      // >0: no thrust this turn; ==4: shield active (mass x10)
  bool boostUsed;  // this pod has fired BOOST
  float lastRot, lastThrust;  // its move last turn, for the network (thrust 0 and lastTouch when unknown)
  bool lastTouch;             // last turn's speed change does not fit a move of its own: it touched a pod
  bool lastMasked;            // inside a rollout: the network is not shown its own predictions as last moves
};
struct State {
  Pod p[4];               // 0,1 = team 0; 2,3 = team 1
  int timeout[2];         // turns left before the team times out
  bool teamBoostUsed[2];
  int turn;
  int winner;             // -1 while running, else 0 or 1
};
inline double normAngle(double a) {
  while (a >= 360.0) a -= 360.0;
  while (a < 0.0) a += 360.0;
  return a;
}
inline double angleDiff(double a, double b) {
  double d = b - a;
  while (d >= 180.0) d -= 360.0;
  while (d < -180.0) d += 360.0;
  return d;
}
// floor and trunc through integer casts: exact for game coordinates, and no libm call without SSE4.1
inline double truncFast(double v) { return (double)(long long)v; }
inline double roundHalfUp(double v) {
  double t = truncFast(v + 0.5);
  return t > v + 0.5 ? t - 1 : t;
}
inline bool boostAvailable(const State& s, int i) {
  return g_sharedBoost ? !s.teamBoostUsed[i >> 1] : !s.p[i].boostUsed;
}
inline int resolveThrust(State& s, int i, int thrust, int kind) {
  Pod& p = s.p[i];
  int t = thrust;
  if (kind == K_BOOST) {
    if (boostAvailable(s, i)) {
      s.teamBoostUsed[i >> 1] = true;
      p.boostUsed = true;
      t = BOOST_THRUST;
    } else {
      t = MAX_THRUST;
    }
  } else if (kind == K_SHIELD) {
    p.shield = SHIELD_TURNS;
  }
  if (p.shield > 0) t = 0;
  return t;
}
inline void addThrust(Pod& p, int t) {
  if (t == 0) return;
  p.vx += std::cos(p.rad) * t;
  p.vy += std::sin(p.rad) * t;
}
struct TrigTable {  // cos and sin every 0.1 degree, interpolated: within 4e-7 of the exact values
  double c[3601], s[3601];
  TrigTable() {
    for (int k = 0; k <= 3600; ++k) c[k] = std::cos(k * 0.1 * DEG), s[k] = std::sin(k * 0.1 * DEG);
  }
};
inline const TrigTable g_trig;
inline void applyMove(State& s, int i, double rot, int thrust, int kind) {
  int t = resolveThrust(s, i, thrust, kind);
  Pod& p = s.p[i];
  rot = std::max(-MAX_ROT, std::min(MAX_ROT, rot));
  p.angle = normAngle(p.angle + rot);
  p.rad = p.angle * DEG;
  p.lastMasked = true;
  if (t == 0) return;
  double x = p.angle * 10;
  int k = (int)x;
  double f = x - k;
  p.vx += (g_trig.c[k] + (g_trig.c[k + 1] - g_trig.c[k]) * f) * t;
  p.vy += (g_trig.s[k] + (g_trig.s[k + 1] - g_trig.s[k]) * f) * t;
}
inline void applyCommand(State& s, int i, double tx, double ty, int thrust, int kind, bool firstTurn) {
  int t = resolveThrust(s, i, thrust, kind);
  Pod& p = s.p[i];
  if (tx == p.x && ty == p.y) return;  // the referee skips rotation and thrust
  if (std::fabs(angleDiff(p.angle, p.rad / DEG)) > 1e-6) p.rad = p.angle * DEG;  // angle was set from outside
  double a = std::atan2(ty - p.y, tx - p.x);
  constexpr double maxRot = MAX_ROT * DEG, twoPi = 2 * PI_D;
  if (firstTurn) {
    p.rad = a;
  } else {
    double da = std::fmod(a - p.rad, twoPi);
    double d = std::fmod(2 * da, twoPi) - da;  // wrapped to (-pi, pi]
    if (d > maxRot) p.rad += maxRot;
    else if (d < -maxRot) p.rad -= maxRot;
    else p.rad = a;
  }
  p.angle = normAngle(p.rad / DEG);
  addThrust(p, t);
}
inline double collideTime(const Pod& a, const Pod& b) {
  double px = b.x - a.x, py = b.y - a.y;
  double pl2 = px * px + py * py;
  if (pl2 <= POD_RSQ) return 0.0;
  double vx = b.vx - a.vx, vy = b.vy - a.vy;
  double dot = px * vx + py * vy;
  if (dot > 0) return 10.0;
  double vl2 = vx * vx + vy * vy;
  if (vl2 == 0) return 10.0;
  double disc = dot * dot - vl2 * (pl2 - POD_RSQ);
  if (disc < 0) return 10.0;
  return (-dot - std::sqrt(disc)) / vl2;
}
inline void bounce(Pod& a, Pod& b) {
  double nx = b.x - a.x, ny = b.y - a.y;
  double dd = std::sqrt(nx * nx + ny * ny);
  if (dd == 0) return;
  nx /= dd;
  ny /= dd;
  double rvx = a.vx - b.vx, rvy = a.vy - b.vy;
  double m1 = a.shield == SHIELD_TURNS ? 0.1 : 1.0;  // inverse masses
  double m2 = b.shield == SHIELD_TURNS ? 0.1 : 1.0;
  double force = (nx * rvx + ny * rvy) / (m1 + m2);
  if (force < 120.0) force += 120.0;
  else force += force;
  double ix = -nx * force, iy = -ny * force;
  a.vx += ix * m1;
  a.vy += iy * m1;
  b.vx -= ix * m2;
  b.vy -= iy * m2;
  if (dd <= 800.0) {  // push overlapping pods apart
    double e = -(dd - 800.0) / 2 + 0.00001;
    a.x -= nx * e;
    a.y -= ny * e;
    b.x += nx * e;
    b.y += ny * e;
  }
}
inline bool segmentHitsCp(double x1, double y1, double x2, double y2, double cx, double cy) {
  double dx = x2 - x1, dy = y2 - y1;
  double px = x1, py = y1;
  double pd2 = dx * dx + dy * dy;
  if (pd2 != 0) {
    double u = ((cx - x1) * dx + (cy - y1) * dy) / pd2;
    if (u > 1) {
      px = x2;
      py = y2;
    } else if (u > 0) {
      px = x1 + u * dx;
      py = y1 + u * dy;
    }
  }
  px -= cx;
  py -= cy;
  return px * px + py * py < CP_RSQ;
}
inline void checkCheckpoint(State& s, const Track& tr, int i, double x0, double y0) {
  Pod& p = s.p[i];
  if (p.passed >= tr.total()) return;
  if (segmentHitsCp(x0, y0, p.x, p.y, tr.x[p.next], tr.y[p.next])) {
    p.passed++;
    if (p.passed < tr.total()) p.next = p.next + 1 == tr.n ? 0 : p.next + 1;  // a finished pod stays on its last checkpoint
    s.timeout[i >> 1] = TIMEOUT_TURNS;
  }
}
inline void step(State& s, const Track& tr) {
  Pod* p = s.p;
  double sx[4], sy[4];
  for (int k = 0; k < 4; ++k) sx[k] = p[k].x, sy[k] = p[k].y;
  double t = 1.0;
  for (int guard = 0; t > 0.0 && guard < 64; ++guard) {
    double first = t;
    int ci = 0, cj = 0;
    for (int i = 3; i > 0; --i)
      for (int j = i - 1; j >= 0; --j) {
        double tc = collideTime(p[i], p[j]);
        if (tc <= first) first = tc, ci = i, cj = j;
      }
    for (int k = 0; k < 4; ++k) p[k].x += p[k].vx * first, p[k].y += p[k].vy * first;
    t -= first;
    if (ci != cj) bounce(p[ci], p[cj]);
    if (t > 0.0)
      for (int k = 0; k < 4; ++k) {
        checkCheckpoint(s, tr, k, sx[k], sy[k]);
        sx[k] = p[k].x, sy[k] = p[k].y;
      }
  }
  for (int k = 0; k < 4; ++k) {
    Pod& q = p[k];
    q.vx = truncFast(q.vx * FRICTION);
    q.vy = truncFast(q.vy * FRICTION);
    q.x = roundHalfUp(q.x);
    q.y = roundHalfUp(q.y);
    if (q.shield > 0) q.shield--;
    checkCheckpoint(s, tr, k, sx[k], sy[k]);
  }
  s.timeout[0]--;
  s.timeout[1]--;
  s.turn++;
  if (s.timeout[0] <= 0) s.winner = 1;
  else if (s.timeout[1] <= 0) s.winner = 0;
  else
    for (int k = 0; k < 4; ++k)
      if (p[k].passed >= tr.total()) {
        s.winner = k >> 1;
        break;
      }
}
inline void initState(State& s, const Track& tr, bool swapGrid = false) {
  std::memset(&s, 0, sizeof s);
  double dx = tr.x[1] - tr.x[0], dy = tr.y[1] - tr.y[0];
  double dd = std::sqrt(dx * dx + dy * dy);
  dx /= dd;
  dy /= dd;
  static const double mult[4] = {500, -500, 1500, -1500};
  for (int k = 0; k < 4; ++k) {
    double m = mult[swapGrid ? (k + 2) & 3 : k];
    Pod& q = s.p[k];
    q.x = roundHalfUp(tr.x[0] + dy * m);
    q.y = roundHalfUp(tr.y[0] - dx * m);
    q.vx = q.vy = 0;
    q.angle = 0;
    q.rad = 0;
    q.next = 1 % tr.n;
    q.passed = 0;
    q.shield = 0;
    q.boostUsed = false;
  }
  s.timeout[0] = s.timeout[1] = TIMEOUT_TURNS;
  s.winner = -1;
}
inline double progress(const Pod& q, const Track& tr) {
  double dx = tr.x[q.next] - q.x, dy = tr.y[q.next] - q.y;
  return q.passed * 50000.0 - std::sqrt(dx * dx + dy * dy);
}
}  // namespace mpr
#include <cmath>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <cstdint>
namespace mpr {
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 88172645463325252ull) : s(seed ? seed : 1) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ull;
  }
  int below(int n) { return (int)((next() >> 33) % (uint64_t)n); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double range(double a, double b) { return a + (b - a) * uniform(); }
};
}  // namespace mpr
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
  double snapP = 0.125;       // chance a mutation snaps to the top bots' moves (-18/0/+18 deg, thrust 0/200)
  double boostW = 2500;       // value of each BOOST still unused at the horizon (top bots save theirs; 0 lost 39% vs v1, 2000-4000 won ~80%)
  double blockOff = 1000;     // blocker: waits this far in front of the checkpoint, toward the enemy runner (top bots: 700-1100)
  double blockStick = 3000;   // blocker: keeps defending the runner's next checkpoint unless the runner is this much closer to it (1500 in v4; 3000 = 57.5% vs it)
  double shieldUseW = 0;      // flat cost per SHIELD in our plan (top bots shield 1-7 times per 100 pod-turns; we did 6-10)
  double robustW = 1;         // weight of a plan's downside against a second (heuristic) opponent prediction (0 = off, 1 = worst case); doubles the cost of an evaluation. 1: 76% vs v4; 0.5: 71%; 2: 60%
  double reactShare = 0.3;    // share of the turn spent re-evolving our plan against the network's answer to it (0 = off)
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
        else if (k == "blockStick") blockStick = v;
        else if (k == "shieldUseW") shieldUseW = v;
        else if (k == "robustW") robustW = v;
        else if (k == "reactShare") reactShare = v;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
};
inline double dist(double ax, double ay, double bx, double by) { return std::hypot(ax - bx, ay - by); }
inline void applyGene(State& s, int i, const Gene& g) { applyMove(s, i, g.rot, g.thrust, g.kind); }
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
  double rescue = P.rescueOf[team];
  double score = std::max(pa, pb) - (1 - rescue) * std::max(pc, pd);
  const Pod& R = s.p[run];
  double toCp = std::atan2(tr.y[R.next] - R.y, tr.x[R.next] - R.x) / DEG;
  score -= P.angleW * std::fabs(angleDiff(R.angle, toCp));
  if (P.speedW != 0) {
    double c = std::cos(toCp * DEG), sn = std::sin(toCp * DEG);
    score += P.speedW * (R.vx * c + R.vy * sn);
  }
  const Pod& B = s.p[blk];
  if (rescue > 0) {
    double dB = dist(B.x, B.y, tr.x[B.next], tr.y[B.next]);
    double toB = std::atan2(tr.y[B.next] - B.y, tr.x[B.next] - B.x);
    score -= rescue * (P.rescueW * dB + P.rescueAngW * std::fabs(angleDiff(B.angle, toB / DEG)));
    score += rescue * P.rescueSpeedW * (B.vx * std::cos(toB) + B.vy * std::sin(toB));
  }
  const Pod& O = s.p[orun];
  int n1 = O.next, n2 = n1 + 1 == tr.n ? 0 : n1 + 1;
  bool first = dist(B.x, B.y, tr.x[n1], tr.y[n1]) < dist(O.x, O.y, tr.x[n1], tr.y[n1]) + P.blockStick;
  int tgt = first ? n1 : n2;
  double fx = first ? O.x : tr.x[n1], fy = first ? O.y : tr.y[n1];
  double ux = fx - tr.x[tgt], uy = fy - tr.y[tgt], ul = std::max(1.0, std::hypot(ux, uy));
  double off = std::min(P.blockOff, 0.5 * ul);
  double blocking = 1 - rescue;
  score -= blocking * P.blockDistW * dist(B.x, B.y, tr.x[tgt] + off * ux / ul, tr.y[tgt] + off * uy / ul);
  double toRun = std::atan2(O.y - B.y, O.x - B.x) / DEG;
  score -= blocking * P.blockAngW * std::fabs(angleDiff(B.angle, toRun));
  score -= P.shieldW * (R.shield + 0.3 * B.shield);
  if (P.boostW != 0) {
    if (g_sharedBoost) {
      score += P.boostW * ((s.teamBoostUsed[team] ? 0 : 1) - (s.teamBoostUsed[1 - team] ? 0 : 1));
    } else {
      score += P.boostW * ((s.p[a].boostUsed ? 0 : 1) + (s.p[b].boostUsed ? 0 : 1));
      score -= P.boostW * ((s.p[c].boostUsed ? 0 : 1) + (s.p[d].boostUsed ? 0 : 1));
    }
  }
  score -= P.timeoutW * std::max(0, 25 - s.timeout[team]);
  score += P.timeoutW * std::max(0, 25 - s.timeout[1 - team]);
  return score;
}
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
struct Search {
  Track tr;
  EvalParams P;
  Rng rng{12345};
  Plan mine{}, theirs{};
  bool haveMine = false, haveTheirs = false;
  const Plan* alt = nullptr;   // second opponent prediction while evolving our plan (robustW > 0)
  double scoreRobust(const State& root, const Plan& cand, int team, const Plan& other) {
    double s1 = team == 0 ? score(root, cand, other, 0) : score(root, other, cand, 1);
    if (!alt) return s1;
    double s2 = team == 0 ? score(root, cand, *alt, 0) : score(root, *alt, cand, 1);
    return s1 + P.robustW * std::min(0.0, s2 - s1);
  }
  long evals = 0;
  double lastScore = 0;
  std::function<void(const State&, int, const Plan&, Plan&)> model;
  double score(const State& root, const Plan& t0, const Plan& t1, int team) {
    ++evals;
    State s = root;
    const Plan& own = team == 0 ? t0 : t1;
    double cost = 0;
    for (int d = 0; d < DEPTH; ++d) {
      cost += P.shieldUseW * ((own.g[d][0].kind == K_SHIELD) + (own.g[d][1].kind == K_SHIELD));
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
  const Plan& think(const State& root, double budgetMs) {
    double oppShare = P.oppShare;
    Timer clk;
    evals = 0;
    for (int t = 0; t < 2; ++t) {
      int run, blk;
      roles(root, tr, t, run, blk);
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
    alt = P.robustW > 0 && model ? &h1 : nullptr;  // the heuristic opponent as the second opinion
    bool react = model && P.reactShare > 0;
    mine = evolve(root, 0, theirs, mySeeds, myN, clk, clk.ms(), react ? budgetMs * (1 - P.reactShare) : budgetMs);
    if (react) {  // predict their answer to the plan we found, then improve ours against that
      model(root, 1, mine, theirs);
      Plan seeds2[2] = {mine, mine};
      model(root, 0, theirs, seeds2[1]);
      mine = evolve(root, 0, theirs, seeds2, 2, clk, clk.ms(), budgetMs);
    }
    alt = nullptr;
    haveMine = haveTheirs = true;
    return mine;
  }
};
}  // namespace mpr
namespace mpr {
constexpr int N_IN = 22;
constexpr int N_ACT = 7;                 // per pod
constexpr int N_OUT = N_ACT * N_ACT;     // Q[runner action][blocker action]
inline Gene actionGene(int a) {
  if (a == 6) return {0.f, 0, K_SHIELD};
  return {(float)((a % 3 - 1) * MAX_ROT), (int16_t)(a < 3 ? MAX_THRUST : 0), K_THRUST};
}
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
constexpr int N_FULL = 48;
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
  double pm = progress(s.p[mate], tr), pi = progress(P, tr);
  f[k++] = pi >= pm ? 1.f : 0.f;
  f[k++] = boostAvailable(s, i) ? 1.f : 0.f;
  f[k++] = (float)(tr.total() - P.passed) / 24.f;
  f[k++] = s.timeout[t] / (float)TIMEOUT_TURNS;
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
  bool known = !P.lastMasked;
  f[k++] = known ? P.lastRot / (float)MAX_ROT : 0.f;
  f[k++] = known && !P.lastTouch ? std::max(0.f, std::min(650.f, P.lastThrust)) / MAX_THRUST : 0.f;
  f[k++] = known && P.lastTouch ? 1.f : 0.f;
  f[k++] = known ? 0.f : 1.f;
}
constexpr int BC_ROT = 3, BC_KIND = 4, BC_OUT = BC_ROT + BC_KIND;
inline float bcRotCenter(int b) { return (float)((b - 1) * MAX_ROT); }
inline int bcRotBin(double rot) { return rot < -MAX_ROT / 2 ? 0 : rot > MAX_ROT / 2 ? 2 : 1; }
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
#include <algorithm>
namespace mpr {
constexpr int GAME_MAX = 8;
inline float solveGame(const float* A, int m, int n, float* x, float* y, int iters = 150) {
  int bi = 0, bj = 0;
  float maxmin = -1e30f, minmax = 1e30f;
  for (int i = 0; i < m; ++i) {
    float mn = 1e30f;
    for (int j = 0; j < n; ++j) mn = std::min(mn, A[i * n + j]);
    if (mn > maxmin) maxmin = mn, bi = i;
  }
  for (int j = 0; j < n; ++j) {
    float mx = -1e30f;
    for (int i = 0; i < m; ++i) mx = std::max(mx, A[i * n + j]);
    if (mx < minmax) minmax = mx, bj = j;
  }
  if (maxmin >= minmax - 1e-6f) {
    std::fill(x, x + m, 0.f);
    std::fill(y, y + n, 0.f);
    x[bi] = 1;
    y[bj] = 1;
    return A[bi * n + bj];
  }
  float rx[GAME_MAX] = {}, ry[GAME_MAX] = {}, sx[GAME_MAX] = {}, sy[GAME_MAX] = {};
  float cx[GAME_MAX], cy[GAME_MAX], u[GAME_MAX];
  auto strat = [](const float* r, int k, float* out) {
    float s = 0;
    for (int i = 0; i < k; ++i) s += r[i];
    for (int i = 0; i < k; ++i) out[i] = s > 0 ? r[i] / s : 1.0f / k;
  };
  for (int t = 1; t <= iters; ++t) {
    strat(rx, m, cx);
    strat(ry, n, cy);
    float v = 0;
    for (int i = 0; i < m; ++i) {
      u[i] = 0;
      for (int j = 0; j < n; ++j) u[i] += A[i * n + j] * cy[j];
      v += cx[i] * u[i];
    }
    for (int i = 0; i < m; ++i) rx[i] = std::max(0.f, rx[i] + u[i] - v);
    strat(rx, m, cx);
    for (int i = 0; i < m; ++i) sx[i] += t * cx[i];
    float w = 0;
    for (int j = 0; j < n; ++j) {
      u[j] = 0;
      for (int i = 0; i < m; ++i) u[j] += cx[i] * A[i * n + j];
      w += cy[j] * u[j];
    }
    for (int j = 0; j < n; ++j) ry[j] = std::max(0.f, ry[j] + w - u[j]);
    strat(ry, n, cy);
    for (int j = 0; j < n; ++j) sy[j] += t * cy[j];
  }
  strat(sx, m, x);
  strat(sy, n, y);
  float val = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) val += x[i] * A[i * n + j] * y[j];
  return val;
}
}  // namespace mpr
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace mpr {
struct Dense {
  int in = 0, out = 0;
  std::vector<float> W, b;  // W[o * in + i]
};
inline float leaky(float v) { return v > 0 ? v : 0.01f * v; }
struct MLP {
  std::vector<Dense> L;
  int inputs() const { return L.empty() ? 0 : L.front().in; }
  int outputs() const { return L.empty() ? 0 : L.back().out; }
  void init(const std::vector<int>& sizes, uint64_t seed) {
    L.clear();
    uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto rnd = [&]() {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      return ((s >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1;
    };
    for (size_t k = 0; k + 1 < sizes.size(); ++k) {
      Dense d;
      d.in = sizes[k];
      d.out = sizes[k + 1];
      d.W.resize((size_t)d.in * d.out);
      d.b.assign(d.out, 0.f);
      double lim = std::sqrt(6.0 / d.in);  // He-uniform
      for (auto& w : d.W) w = (float)(rnd() * lim);
      L.push_back(d);
    }
  }
  void forward(const float* x, float* y) const {
    thread_local std::vector<float> bufA, bufB;
    size_t width = 0;
    for (auto& d : L) width = std::max(width, (size_t)d.out);
    if (bufA.size() < width) bufA.resize(width), bufB.resize(width);
    const float* in = x;
    for (size_t k = 0; k < L.size(); ++k) {
      const Dense& d = L[k];
      bool last = k + 1 == L.size();
      float* out = last ? y : (k & 1 ? bufB.data() : bufA.data());
      for (int o = 0; o < d.out; ++o) {
        const float* w = &d.W[(size_t)o * d.in];
        float acc = 0;
        for (int i = 0; i < d.in; ++i) acc += w[i] * in[i];
        acc += d.b[o];
        out[o] = last ? acc : leaky(acc);
      }
      in = out;
    }
  }
};
// Blob: "MPR1", u8 layers, (u16 in, u16 out) per layer, then per layer:
inline std::string quantize(const MLP& m) {
  std::string b = "MPR1";
  b.push_back((char)m.L.size());
  auto put16 = [&](int v) { b.push_back((char)(v & 255)), b.push_back((char)(v >> 8)); };
  auto putf = [&](float f) {
    char c[4];
    std::memcpy(c, &f, 4);
    b.append(c, 4);
  };
  for (auto& d : m.L) put16(d.in), put16(d.out);
  for (auto& d : m.L) {
    for (int o = 0; o < d.out; ++o) {
      float mx = 0;
      for (int i = 0; i < d.in; ++i) mx = std::max(mx, std::fabs(d.W[(size_t)o * d.in + i]));
      float sc = mx > 0 ? mx / 127.f : 1.f;
      putf(sc);
      for (int i = 0; i < d.in; ++i) {
        int q = (int)std::lround(d.W[(size_t)o * d.in + i] / sc);
        b.push_back((char)(int8_t)std::max(-127, std::min(127, q)));
      }
    }
    for (int o = 0; o < d.out; ++o) putf(d.b[o]);
  }
  return b;
}
inline bool dequantize(const std::string& b, MLP& m) {
  if (b.size() < 5 || b.compare(0, 4, "MPR1") != 0) return false;
  size_t p = 4;
  int n = (uint8_t)b[p++];
  m.L.assign(n, Dense());
  auto get16 = [&]() {
    int v = (uint8_t)b[p] | ((uint8_t)b[p + 1] << 8);
    p += 2;
    return v;
  };
  auto getf = [&]() {
    float f;
    std::memcpy(&f, b.data() + p, 4);
    p += 4;
    return f;
  };
  for (auto& d : m.L) {
    if (p + 4 > b.size()) return false;
    d.in = get16();
    d.out = get16();
    d.W.resize((size_t)d.in * d.out);
    d.b.resize(d.out);
  }
  for (auto& d : m.L) {
    if (p + (size_t)d.out * (4 + d.in) + 4 * (size_t)d.out > b.size()) return false;
    for (int o = 0; o < d.out; ++o) {
      float sc = getf();
      for (int i = 0; i < d.in; ++i) d.W[(size_t)o * d.in + i] = (int8_t)b[p++] * sc;
    }
    for (int o = 0; o < d.out; ++o) d.b[o] = getf();
  }
  return true;
}
inline const char* kZ85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
inline std::string z85Encode(std::string bytes) {
  while (bytes.size() % 4) bytes.push_back(0);
  std::string out;
  for (size_t i = 0; i < bytes.size(); i += 4) {
    uint32_t v = ((uint32_t)(uint8_t)bytes[i] << 24) | ((uint32_t)(uint8_t)bytes[i + 1] << 16) |
                 ((uint32_t)(uint8_t)bytes[i + 2] << 8) | (uint8_t)bytes[i + 3];
    char c[5];
    for (int k = 4; k >= 0; --k) c[k] = kZ85[v % 85], v /= 85;
    out.append(c, 5);
  }
  return out;
}
inline std::string z85Decode(const char* s) {
  int rev[128];
  for (int i = 0; i < 128; ++i) rev[i] = 0;
  for (int i = 0; i < 85; ++i) rev[(int)kZ85[i]] = i;
  std::string out;
  size_t n = std::strlen(s);
  for (size_t i = 0; i + 5 <= n; i += 5) {
    uint32_t v = 0;
    for (int k = 0; k < 5; ++k) v = v * 85 + rev[(int)s[i + k] & 127];
    out.push_back((char)(v >> 24));
    out.push_back((char)(v >> 16));
    out.push_back((char)(v >> 8));
    out.push_back((char)v);
  }
  return out;
}
}  // namespace mpr
inline const char* NN_WEIGHTS =
    "o&a.x126.F0h6s+0h6q%0h%0^jrcus&>Crw0S){f5+KAm2)yIk@R#@&0T[#q0S@F(2nK@J1o5f0@@ix90S@h(@r5YBjpJM0E*MhQ0qdUM}#N051P[/b0qDB2"
    "VJdD->l[Dk4Glzb2ls&4@q=u&0rAe#1lO2PjPCM[0#y}r0002#00991@@r300ri61}*[.f@R(=&0S-o30000000i91@@A60%7xhRjP3.w^shfi}3N]u1{s@b"
    "}#DY}@@8&)@sGen@@S\?{0@JGj}U{G{{6p:42MK8-{W+8+jo$PCfXlj70%64)0SJo9%nR#$0@@o2F=1jb[EkmV{5a:&0SSi2@S8V)0rrl20ma{Jj+j1GgCeOb"
    "0S&lb0r8(%1o5i000994%nA02@@r8$@q=SV@qM=%@@0020S-u60I-2<jotM-]vB0y000600r-x2@@r91@@Af0Ul<BT{XV*%4fV!m0S-o2@@rc400i2@@-%n7"
    "jOvH8FG+s0%nAbR1Q1M35bVqb1P52[X#EK\?73FM^2){e8]-s{<1onx3008{004VaNjob&Z9Pm2000930009c1@@A9100001J^{mg@08:41{1-90rr9100000"
    "0S-o20iFysjHAY6\?R[za0%1P#2\?*3>[-P#T@ql={3jv4t2(Tk>0{Ljk90ciA0#7871]W#)@5(q#jOZ0y/HtYj0Ta08]zXpZ1Pnc1@R#{b3i*q70@@i4}6aGf"
    "-v7()01EGZ1onMb0ug%{jp8u8}T:)>0rh%20SR@}@Sq@%00000@SAl300i5{0n8=d90low@@h(%1PFP90hfD&jB[ar[BN4&2k/Li6bnd-}tyD<0SJb!)G9Z)"
    "}tR0h5=R$f2Lnc50p/j3{5t0&@Q%zJjoiF]@$>>r5CcU0}Upl}[ce^/}VWfh=WU%h<OmS1<P$R(1oF(a0S&A4}VD-{1=+kojPT%A=/>Af000l50rAl3@@A60"
    "00000E(eCX2)p$M@r962000300rJi20rAc1@#3[wjPCJZ<M#%D004PE0rJx800if20rr604HJsu0rsR{}#u^00rrf20rSl21oeD80QEZLjok=.2snp<1n>Mr"
    "2[3!l}#uY%@@I(\?4ikn\?1ma:C)GiN(2lT#f@S&Yb0@9oe2K>M&jotrI!A3Yi0rrOI@Sz##0S&u50Sz##>/8/w[A+sF0%wY90rr920@&b#@R#($0g$HyjixwV"
    "0RlH27YS4h6OcwL0rh(>2j.L(02kh31QC>l{XTQ@}3^Mq0M@2+n=nFU%nIJ0jPk-&{XBZ!0rn.&@@rc3@q=&$000310%nJ500001%9[q.3Jp5b0q=Y}@@8%2"
    "0CeMijLYNynpQTw}$[g/1][q\?]-bi}@Sh]]2[EEz}VNoe2!xo{<nO&U@qMY#0$L8d{9#^6joL&*apwnJ0@@D400i8#0rr61009902N\?yI2N2.b00if3@@A91"
    "00i91@@A5#%c\?Zzj+i$kfaXGJ0rI&%%nr94@@r31@@A600rJo400030@}(<00%nJ50@@r30SSx60R{Q8jos:Q!FOw!0rro500ic20rAi20002#F3HA=3hK#7"
    "1owM80rr91000310S-o1%nDwXjp8l2F0H3q0%5A600JJ83hs]g@@Sr3@pPNW1o5Ji}uDGQ2)g2800JA80Tfhy1bm*{jPCD.@S@M8008##009300000000000"
    "000000000101BFY]A1{>%nJ600rAi20nSFAjs.o5}k%5J3\?K^({aCH#[^uib}srnv}0n2e0ql{n{@4Z::]zby{wCp2}SX<[10{urjPLGU}TBK=0003100000"
    "0000000000gjGGp0rJra@Si3000000000000000008@Pqjosp[925FN0Sy(+0SS2}0%FM30@Sx6k.fT5[FNio1p1-80@-l1@@z#$}2Z^41C5ZYjB[Ew>#4=:"
    "@@Ipx6zq1%4f=!g0@Sn]2>m()008*8@s[n}<oU\?/1l/jL%nJSn33G49jOv.4\?u:k-2M%$[0}uS}@r0930@-A3@^+j]]:I0B@1xnr{w>r{1p1=8@qlW20D1eH"
    "jB$us8Y^Gn}tK\?u^^cx7}1<T.0Q/*\?!*A3:}#tBE1KN\?R4eZ4#0R46a{wUK729DvKj2l}/140Dm1pM\?r6-OvI6:1Wt2MtCi]zFs$1QUd@2!*Gu[bqUO1QT8*"
    "4F\?QY6s^DkjHNe+.Iem20Si*I[.rTi{520$@p/{]7YA$q1P!+({rJ\?SE<=9l@\?Yt<}#)Gm1w[iRjP23\?^+a:0}#l-@@SS=q@Sri60SSl1F:1i=2NR4y}$yc)"
    "1{1=900h]@1]wr40nJwvjqlrZ{#\?>Z2(Sb16XH}q2N*>t0@q^7}$*l@1oXR#1YS[9-ol[10MidW]8]j43C>@5joK%t3&Zk9@@JVb@@rl7}t*P#00i923v-40"
    ")f9l}1oF-c@q=(#00962%nA31@@]Fijq[7%(&(RU2lGD2Dhd^S{Y\?98@U>!sb(X2C1oOG9@OEzc0rAf30}SIU!CX\?l)0f!.joch.82Bsm0SSa*0TnD60rrc2"
    "0q#]@FI#9&@Q*G/1]XD3@Si600@@o2}#uS@0l@4Ejnm-d^^NUz}$^zn9UvVT@S-951Qj!)[Ny}Y@tN6C0dOnQ)H6j8{4)RE0$<n^]nm#djOx6{S00bg}#o^$"
    "3K=O4]y&p:@}(Tb2npzDc\?zwf+]^q(}VD*2@@qx^3\?<-a3wB5/jou8@6.}Xe0rixd00064@@r300rAf2@Si9100ic10Fd344GuIg%m(V}0SSe#@x2{5jM46u"
    "2HrzL00hQF3HEuY2M+5d@@A6a4flni0@@x6}-LShF=a3#@#g.k0R(0N{(lpOjoXBb76K^B{w0Z12)fO^}Wa5p]zO)/coqnE>IK}z@R=/#4GMwc{YH0=1[zY("
    "2YuDjjN-E.\?7O*e0rp^@0@z@#00i5#0rrf200h@$0rri30E{v*=J&.n@}#&#00h@{@QK1vjoklW<%y6d@@i7{@@Sr70%5x50SJc2yK)h*1]XCY}t*J{0@&l2"
    "0SA91000010oh1vjO/D/(+9h@0@^=z1]v@%}#V-%0riuc/{@DY1pbMt5=hLf1n&l800SYg@Sr/m3yl%Pjo)gc0!yvK0S\?S<@QG{>1oFG40r8{0@qV/@%nJ90"
    "@R#S(0S-l1%nA910%5Mc1p\?Umjo=Gt^yxqS@SAX9@TO(d}2yr@@R^96m(JjHa$wfp-SlqG]8!<>@@S910@@r3@R@wNjoLcy[9o(M00iaJ0r&G700930@@A2$"
    "vjifn00}/&1onx2@qV<0@S8/{0ri300&p&cjpo:@T:6*y0@}9Q@rT2k}2px#@S933oRi6W1PyrgapO[V2l>nf0SJc10rI&(@u$<GjobDSk(pY-@Sr7>0TwJ5"
    "@@J900r903{:MyZ}t(fz2M<#c0@@u600992@@rl70v2t3jC0v+h[w+\?0roRm0Sh(%@qV)0%nJ930@@l3%nJ63}#u^01oeD40qV=%0SA62@TjU2j3JW]8O(jH"
    "1omDV1O9U@]9CG%]ag():xGxc5zS0@1R6h[%nib[@$C!u2Np8i2*hPujP#G56:1Eo000Ac0rAi20SSi20000100032@@A900F7Kr}2yu[0qV-%0SSe#@::Yp"
    "job=U0}#^11PFq+@SR]V@qV&}00&=7UZ+O1[f4Ow[d+{f4/=Rh@%wq%0ri010B5$:jOKH1T0d=H1n(je@or$T{5Co(@rSufg71{F{xVok1OqY%2(jVb2k^J!"
    "2)6=2}UEM[jp}t}Tn.[{1oRD/{YISk}s$9{@@r5%0s4#\?1{a@c27D+u4I]AO@q=/#1nRG.{)u}tjoEkd]1j\?#@qOQC@snz(2MsL#0S8@<cv+.z{7d#G&$z1]"
    "@@8%81pt2a0r95%})AW1jnF=8<RPDc@p#Q96x[6O!Fe8.0TPzv7s&&^0qlcU07(Apbj(&T3>h\?i2J1$G{#bPojp\?L&<pzg*}t=9e2mx@a@Sr940qV*40seP7"
    "@SAl9@@r98@@J8#%nzY@1o58[15[+LjLw*kk7SIf@rk1@2ij[y1ptqg0TX$)3hjM5@r0ld7lnhtVL-fm0pQr$2Ld#*0pDgKjp*uLkQcg8@SE/A]-=0a@S-c1"
    "1nA9b&%=jZ@r<5k)Ljwf&@YhY@@-/i2(sSd1<Rt3jokM(at<h$0093a@S8<2@S00300990}Vc=$0SJf4@^^8--02/n%n8S)%nh(%@HKN7jO\?LmK[[$U@r4SF"
    "[De1T00h<1015(p3iP${3<nD->Mg$>]Aki{@@\?S+0%O2={cBiFjp%^diqh.(693Mf7wk=(].[)\?@}3Z9WG>B[0@0Vt}HGgE7w!750@IA]0@.&[%hr(GjPCY!"
    "5b#Rf008@%000000000000000FEq8K@S8V[@S90000000009300000000&\?>jotJ-F.p25}VlP@0096200Au60r92#0SJl6@S002{X>f$0@-l1@}(V{@S0Vq"
    "0Ed7$jNyJ6aaBp>0TsI+0R=:0{wK:*0q(S]}x=mz1pCXd]aRGi(&Wy>0@J8#0TGep@*Z=Gjo$f=TrN&F@@J-02j@b#0%5P8@RMW1Zjx8p7w]sr[^CS#00Jl2"
    "%nqu+0@-Dg1[jA(jO6KqFMh{N0rIfI0RDf.@SSGb@R(-}}WTkf}U{=>}346p{5j::%lQo]%mD%e1gUoJjMJ2hgLtR%0}X5f1oYbm2LOD71pCqvhwDED@14--"
    "%m#rQ01F]o2)oq)}$/KQ}vZ3UjotM+>]x:7@@Au400if20T5D40SSe#=)[8#{ZgO-3K3Fe@R#@#009610%ex204of+jO^1>@22Uu1ohQu1nS9l6A[vm@Sq@>"
    "VWSLB0@}QI{v4UM2MjD40S8{0@@&u4@%teqj+i>seBirr00066@@r32@@A9200963}#D/@000010q={1@@Ji2000620rrf20jd6hjo+M+}7I:J}#D@i@$kFh"
    "0@I]%1oeq@.5-WS}Vk]i1Qa-70%eJ21oel1}#N361B>/OjP1BS4\?nQu%nHE61oX&e{wUf@}$yo*FP6:472}3X2NgFm0r@Gb@@\?S*009c1@}0[ejMNHA(!}{A"
    "%nAca@SqY/0q#]#0rA8$0rJc10rAx6}/mfr9r(Jy0%eG5@@A2$@4t8TjqQ%JpAH!@3i#a()-M4]00icl@{Hu>VJeIR0RmJH&o*<.}Uy=%@sd@U68zI1{\?CsO"
    "jNL9V41wxO2N6$:0P$jd3iZOl}#M{3[=OTV2(jOE1mk1Q{X+3<@Tm-Z1n-Yq3T.GRjotM:3X5cU008]}0rJi10SI]}0S&r2%nS8@00993@@8Y/0rrl60rSl3"
    "0SAAf1/=7{j7<49@SAl31oG=0@rnF+[e7-)%nz@{)hD(q[b<vm)Mzc/2l%-1[e8&k}u3%03Sl3{jPLGJ4C}f!008](00001@@i32@@r2$FJ99s1QCzu}Vl<1"
    "@S9000rSl20S-u70cvtljPtDw2*AzJ@@rD70SS614GMFd1o5b}>70mL]:q6B0S&5{@qV&%00if4@R(V[@vHlbjPwgGFGj)\?{5UEy0P9XV1]OM60S8)23<3zh"
    "0SIV[5109m2JVi!}XO/5]-amU{JeDqjJy4R2.rfM@@Dsb4eYV3@%5D5@@8@$2)f]51oO-a}rPQc2M$be@08u/0Twq\?}nz[.jqEdNFKP[p1o=:+%l6Z(2M:@c"
    "%nS32)ev1+}Vu%h%ju#x@SAn#0qWJl@tDaJ0)[Lsj]+G-{ZNo7009300ri30@@r3000031ms+Wl0SAi3@@A91@@r30000000000000YFCjMnZj5[Zjl}#WT:"
    "}V=Z800Af30quY(pZdmj6y#:n3J7Fo0@&Db4/VUk@o0Lv}Ywo:jsz5$fJFR:{xJc#3jdjk2(jP3@}L<:}OhmU2L#*@2law@!Ce>m1p28g0QGpN}uX)mjp8Dq"
    "(WO*30rJYH}$P^[1Q1Yc1]wo7[a(!Y%nq%3}qTdq(K0H>1n-Me}27H32c44vj4:5v%4078{3\?3(7q*^+{xHS)0pZr<2lbL9{YgoXip(zb00i63}UZV$0rr(y"
    "5hZ4WjMsrwwwF9B0r4DQ}sL3&{67x}@qlM)]a.lh@003e}YVB400ir41nSSh}U6*:{m01XjN8yV48rbD@R}uS@@JGd1{a=a0S@J7{vFBX}#MG}2mgts}2*=%"
    "@%nD2@px{&06EIKjPLAX4GDIf0009400i910096100000FD$=C@S902@@Jc1000000ri30000300oqZ4jQhb*@@So61oi]S3Jpwi@@Jc10rr60%nJ930rSl4"
    "@}VZ7%nA2#00Sl10rr620nsQqjP2M/]:R3500mYP0rh=)1][-a000020@S2%0092}0tgOd\?#S]U0p*i(009Ag0y>6{jpP{[::<P@1P(*J2M0P<6.\?Of1nq=@"
    "(*jDF}#45@0RbZ(5cI[k1o5Yi1Nlo}@WydsjoL#L@PTs.@@J8*00AA41owG60rSo341w%u68Gfl4f(.e009600S-8$@@I#}@dko&jojNt&QcZD009Z}0%OV9"
    "@S033%nS90E)Cv!}1K4x}#l^000001008##1onD5%cKtZj5peyF\?l*J}#Rv4@%4xW{XUc@}V)q$7.#Vw3igzN]4I)N[DO*(1@XiO3ICnp4v2@7jok)18YO>U"
    "001by0SJf00ri3000994FRSyI0q(uZ0S&o3@R#@#@@r300rAf1%0w-Xjcp*bq}J-52)#.[dEt[u4\?5gl%m3i>d*b\?D%nJD45Rn9x9[Qt)1Sddt3jcs:@TEiV"
    "j]26.@}DA<0rN&h}WJP52MK/50%w(c776)i@qDM@[^b0>}#=@#0@-i4@Sro6093amjoU]Y{RD$h0%one2m7ng{X+f{@}#)1@Z[eW5*N[e)fAy*{5+i[0ShV)"
    "2MK#c16t*OjQ7#&2]-<A00hY}0rSr200930000000SSi3000320NVF>\?tv3G00000000300qC^Fj1iYlF.Isx1P)nq/!SBH}v-(m1l6k0g4D$z{4OZ)[F@Lh"
    "1[@c1@Sp<O\?3iAz}[klrjoV5\?k}Ke(9@N>H}**fH)/@vY@s/qqujNLt1O\?@<2S<pm]aIlh0r9=l@rBeC1RP8@jPhUpmNv7R3i*!a0oUA@[=F:<@%5U#-(&}1"
    "bLU%{&$qT>4GlLc]-L9[0%Ft[}OsoQj]:*^11acI008@>0Sq@$@q(]#0r93027V^m4f3290%eD6}#D-$00z#@@@Ac3@]e8OjCDD&X}74.}uumF}s%N){5s{)"
    "@qVY>0seS6@Sh/\?3de]y}ub:Z0}UpJ0ric72nB^%jo^4/..}-10@-k.@SJJh}${r{@RM<7[e:je4=^KdFDtVF@@z#$@@S2%0TwG40d+Cwjp6Tr3!UBN1n/jp"
    "2P%0l1]wD20TeA54dX[t4DCwI/!ABE}Vu%4@@}@[2(sP60\?S]RjB)$zd^v{50SRZ00q==@@@A610000000Jo400i910Sz-[@R(($@@So40rAo60W>^rjoDe\?"
    "0#}WG}U#ck3<ct30pY{(0Tw@b/B!\?\?4fnJS8t9X7}ul{90SSu43igOx4zKz\?jo=2@1N#G>@S0/k0TYbb1o!]d0rh/]^+xud/zEiH!F5h\?0S&u2@@\?/>2laYh"
    "2.l%)jo>VY{w+c)@S8JY}#((%00i940SJi01onJ4@@Ai00d+%:1owJ70SSl3@@Af60mQi/jtErE7xB:v3JLbrj]aM@0SSl0@sP2pjnP->1PFb{6GVl*0QP%V"
    "1Fyt2[=emN{xAI+j+j7v7x%)s00h)c@Sq@$0SSf00r8@$@@Jf2008@#0}Mvh@o%^!0rJc00rh@%0*SE+jPk1Bhb%UV0rAuc4FP@a]zwN&0SAb%FYDuH{Weds"
    ">>$X-0SA610rqY)2lCbi1g<9yjgfL\?gsCW[<pb<S5csiG{6{W26AX*l.$<3p}VcT2<(3yu5cAvz@$k\?v2poiI@Uf{njqd-dTSEVz0rs\?:0rz]}1][Y60rii1"
    "@@rc5}tHo(2>9s^!+f4y01]zj}uc6P}6PmNjobu-%nh(%%nI\?B@@&Dd000950ri90cVHgL@SoBJ0@z#$0SSl2@@A2#@q=&{@%lm{jo&vit7lRO0%gdA0%eu0"
    "@09l4@Rci*s1KK{2l6{-<]be-0@-f40SIr=2(Kn]0h%r>jPd4G<jT.A@RBTO@0}#}9S=lB1O.u.FWDG*a%v4R/adxG5c-7k0$0=^@q)032lwVfjok:17:/p-"
    "@@A1G00Jo30S&r30ri5#Z[6#L}Uf{R}Vu=#0SJc10S&x5@@A5$0oKI-jPU-+4F{b9008]$009610SJf2000000rr91000310d=]%1PwA400000@@A600jHs>"
    "i$2THtVY.N5c=r/}1RnZ%mM9Q00@@n3l[y4(<L&yd1qx{{U2MI@Jkq[]:RD5@\?e(BjojKy&oHyz0r8oW%nh(@@r0000r031msMP[4fcv%0%wP900000008@$"
    "@S9620)f9}j+iRsgzo6L0rAfj0R=V$1onA50r901@@h/@000600}#u*1oXP6%nhS)0rJu409$fEi$b(closw=3KdKr2GdS53K4dF0S-DbF.Xcn{xiaZ[Bb&p"
    "4fch6}ulY}68Tf-4xx:xjNr<\?-BPhJ3Jzeo]X/cN2[D!h0$b4=4c!A61n@f26@*(5]pL%L0d\?MZ@%!/o2d>0HjB]9@{up-K4*iaC/Dg.z@t(1s@$1U)\?1Su1"
    "0%5039hx/p5BGn]2o1/:3h&PA[xn:ajodC&aC4N11o!8(1Ku@B@00c51O\?{0{-74z{xHV*[ElYJ3JZFc2keu5@@ib}016XYjoRIi@sn=a0%N./0%Ffj[:)m<"
    "}u)C%OugGt<*B5%6..v${wC6\?}tZf*}uco<0ixuJjs8pW>yI4a}#=pd0qMV%1O@i2@}Du/}0m>:0rrMb4fbq:2(:(a00zM!@Tnn%0ah+4jotD+2MsV5@q.xR"
    "=i>3P0r002@@Jk$]As%)0rAb%{wL0>@@r310}#&#1onn#@Eb*zjB)!w2lj=80rz^6@Sh/]@@r3100030@@r620rrc20Cz-/2kFl100Jr5%nJ2$@>.lojH2#q"
    "3v-fW%nRQI2(0Y.1P[]b0rJr22kwk%0rrb#06GD#j{Abh0Vy\?m0rR&/}+ty\?jE!X:d$m<c0r8B00{Y%+}#V)10rJi53JQ8i0%!Yd}#(%h0%wS7009300SJ2]"
    "@<%MGjB[sIjm@WP008)d%nq]{@S0020098#@Sr9100930@@@Ah0rAf30SJc10SSA91zBhVjJ=NO+lXjJ7ykJ!jE$400@q#<0}ucATNkxP@@>1MhUNgN@}##&"
    "00hi#ciBhK03{==joJp]>p3Jn%npBr0SzG&3J*Un@@R&@3v@I*2{w^O76BHl1o!-800IS&@R#Y]@Pg<+jNLoOQtc)>2)bUU6-n{I@@@Jc@@&D800TeH}U/Hz"
    "}3NJn{wU9{@oT:<0%n=i17EOSi#p@>4uk>Y2ma9e[uiuk3hjMg@$>hp2jzu]4GMbm{XBNP1]eb=}UpJ@6:ln\?bn<n>jQ7=K%k<mX009f40rA8#0SSi200031"
    "0@-l2@@r300{CapbMv&C0r@G600i60@&n]ei#nSBZAbM@3Imkt^sARn2)*+h2NQRR9Pl}+]8*r:{6PU182J*]2l)u#}Uy3{2<Ee>joCDV)EgbR0rr930SSf4"
    "0%eJ70rr8#@qDV@009600dPb74fuRh0S&u40%5x2%17P2jPc+^/du!03iyQ&@T/d}5cR!91{j!%g{O#](L$[l2p/rs0%O/d}#=A!{66<Q{0m$0jPCG+3K^1n"
    "@@z#@009920rr6000030NsUnh@@h/&@q=)00rr91009610rrc100X9zjpqE\?F7sR(2lLtA%j3cj8uo^q0q(P^3gdT02MUON1kRA55DIXf}tZc!0}#%00R&jE"
    "joC/-nvO:60SSo5@}VJ)}#u*1@Si2@/F&Qs0@^}7<]+!!00031@r030@}M-$0{*rajotM*4Dmo^008n{00rr7%nJ920SJe#9[H>=2kLMO6AFgj@S90000ri2"
    "%nI@}03:V]jrs:=6Ol2M@@ahS]ByY!<oU.P@SSJe03RyP@@ru19MyE$kKO{X@$CLo}#Ni31G]9ijOmofFF5Q=1{1.T{#7Sk0o%T>@UC+46!Nw:aU!b#2OC$Z"
    "[=]x#3&HFi0P:pX@q^t(jo6/Ts]ct(1o#g+{pO]Q3<M!t0s5P92eQb{{35mQ()\?Po@qcS&5/zeP{X+l&3V}TfjotcX&SPI>0rvVS1)VA&0@&x4@@Jb@69Gc+"
    "{wtJ{1}Hwa3iyhd@Sz->@@8/{@Wy57jPaBP0#}pu0rmSH@@8/$2lB=60q((@bh:9{@}MuS2(<M00rJf4@@rc30rSl207{&\?jrjIg6YnV!}VvW^]DK.71POR#"
    "1k1z368-jk0S&uc}$nCa6z}\?g0S0-v@0.@U}F/ABjP2-/[^29<@Sh.z0UC+t1o[&90Sz][)X6ZH0TGzP00rr40%5D5@@q##@@q]@0BvqtZkPbUZ3%=Ljj4I@"
    "Z5&i2je\?xSkee72Y}%C]Za+^ake$O+ZsoXNZf9KuZ8OX*Zu]{DZow.7kj+&^Zsn[(ZI1IoY=tjFZl)n$ZuU[Ij..reY<^rgZwuCak5&}GZdv5mi)1qvZ6dTx"
    "kcDT(YxW=-k7P5KZj4UdY%cDeZxaz^Zc[2eYj$[6Zo&$<ZwKwhZg{&BjVI3pj&ko@ZwD#UZ2q9.ZUptgZd^k{ZFV7-j[#P@ZtB2dZd%0sZcikrZsZw*Y(7q*"
    "k1+gFkgh[1Y%7VhY!o#Pj/f)9YO\?tvY^<K(Y\?YP)ZvsQckko:nZGk<gj-v=(jYGccjYaC[ZcnOPj@.N2Z4ybYZkt/hZt=#0Y}0gdZhqN=Y$pBUZbfOQkeXe6"
    "jTri0ZoHy[Zswz3ZkH3<ZiDO6Y!fZEZF+RAZ7JVFjQN+@ZOu=RZuwg.Z89TEkh$KOZ8taoZvjAkZKVbiZXE+$YVxa\?j=aChj\?AizjQk(pZslD]k9/Dujy+9C"
    "Y%#/iZy@U\?jU3WsZp0l!kk$kyY]%YvY)5O5YvomQZW5iUZy@&pZn9jtZu+HaZq4#%ZeASKZlS:JZT2@Nj+Md*khcYlZT8ZmjCcXNZv+x!ZlVrbZs0^oY%OB="
    "Zd*.QZi[OJj(^>[Z36NlZg1#xY{mAij/dVFZl+RcZc4gYZeG6GZn\?KcjYlP=Zqb4KZa$wmZuonmZC%M{Zh<^4k0DvCYXa$LZQLH7ZPy(#ZA5G+Y$q(=ZJfwS"
    "Y-AJgY>TKqj^smfZg-C@Y@yxhj$C/\?Zmu<EZcj0QZcKATY{g)QY@ZCWZ34Cej=e&mjMunj2)gdw@QYdM[^j3:+Q&5@1s\?[h(&NT{+o^R&]{li)XesDn7yZPl"
    "5bvgN4eD114!nP{</]:4\?s*7q@SHZ<)*v4$/-RcH1R*KU}${m1UmzDz5CW:G73CF{1ov3QFbX<G2>dl:{7W2O}u1.T}SX7+<]$L:4!]G8]/Bb<5=A!1.Tmve"
    "7-I]:5bcT^02kRd<[wA3{08yWfWaX}j]tIn@}#(]2ls/8{YZN1{5Li$0rI&<[C-Q!}U6*]{x7v10rSr1@@r305=QX(1onV61P!P8}2{T31{K@c@qlcZ0S@-g"
    "1o5z)4f(!j0Sh%8}T<**T]nL7}uc<02laY1}1<^^0SA66{xP<U@R#*1%nh&<)GiC00}+NT0s5n@0%O-c{Ygo{1PFJ1}UyA<0$<h^)/@Q)0riu9@U7)RjPkMh"
    "]6MhB0MsQa}uu]>[^LB7}27l=4/DIhF+G6J008<a0%[@m0}#=#(hHeN}Sfj5&S/R=)/0mJ@Sh<6[=]3+00:nq008JP}we=f[+&ve3K3wc{6pJ]0T!&a1O&x7"
    "RsL2{2(jYg%niod4HAsq3hs2$3i*C9@pfQ]3b#!#1]4(}}#u<13iQFA1O&l42[dmg(&M+=}0dqB5bgQkjNIyK(mR#t)^PPu}3vu6}vr2{{6/BRFE@g(0TwYr"
    "0T4W0)CX3e8W=&L(&#$R@oGZ6@RWi9>>$q}@0ii5{u\?[T]+u)61{tp<0uH>n&r6bJ=&TdPg7$t%\?t)&)@QZ9Cb(=&}1{K#u[=^5a2kXV0}$*6D-0MLF[cMYk"
    "=)5nl0qcy24d0//3c&zL0o%{[3[JXi*tg)1=Jz+)3{4mMjOXP+1{ahN7Ue3=0rJC=}vAPa3\?iDU=iBKLV/I.>]ACdk[C-dP><n6ybl9ii3HOYs8u{0t0TF@0"
    "{4Xy<[F8n}0pHHg6.hmP]ci@k7-JV4{5k0%Q6gkO6A5a35^E[c9sh9i5c0170pyp484k*n74x-W7xTdF{-GnB1oN6U(*>kC0U2Cd{6RR@@%nGi3*bm$}qzee"
    "6YN%G56emujo>S50u77>!Cm\?}0$a!<4J>VJ@1azY]9Cu$[Fhri00-GC{xggk0#7Ll[Cg#=]zYAF+KlqM[7OuZ)hcrF(<S7S@UMH-1i*xvZYBq@2kO-50U2hh"
    "4a>{>]V)Ty@o>mp5!:Hk8vl/-@j-6!0Sh&<{vFa+01Gk1(*\?.1)D<Px}%K)S@r06d2ltaZ0@-L$1qcHZ<OO6G@oKZZdCj)EjPbu.]8Gm6@@qS)&skzV0S8<2"
    "^HDoj1oUa$0%Yhd[=o{k1{U8i0qE$&4HJ4l0@A06{YpY!0SIY!]aQA)4G)7p%nI<0@Sh(O}uaXp1)#o^1]OA0@SiDl%n0992lj]d@@&Mf}2oBW{XB%^3hjf1"
    "}5o8F00r9a1owHN1PeV3}2HJ%@09kZ1%jrV@q4JjJ%CZ+@@qr!1mMu{0Cin(jQRk&@qVY}2NH!r@@Je$@S8)40T5J9{5a{(1pb59@rq%7{wLf#2Ngec@}>jC"
    "1PnV90@&A4}v9x5}2gl#}#3u%0@-lb2Nghc2lBY8009i60S9l5F=b6s@@h=%0r9oa0r-r7}#V)4@pQo^@}V=]@quJ$3iZLf1]e94}VD($2k[Ma@}MA)000z]"
    "@@hY]1PE(:0SS62}r[Q>{b6oIjM(2Ec\?:}o1%\?Id[L4NB5Rou.3^fzC}%K#K0]sKU*uo{.2kkE$>m)vFg6=5e2lq{G^5DBb)FCn!{5y-%\?Y}>e]vIp{ht^A>"
    "9@l=H{ZEVd1>\?\?X2lpK+7VD5c2n[JG/ApBP4^<0i\?tv]:!..!o5+</*(D4+%0%m152)DsD3/gy]dH(WB7:d&Y9@b:k2K@8-0t+W@90WhR&i<!/jPt@\?<o2>}"
    "anu(R2P:xH%03$z01FZO59:nW4*%JN3geG9<P1P91n-wR6.}t43Jyqf)xVS<}3W67\?$#KA}-28}(jK9A}#39@}2{<e<>P$c2ic7m4=u{-*Uqc7[cm}I3E[+:"
    "@Rc=!&pM*3<<hQ0}{s^84G=O559r%p)*6Ps!B:BD)E{[(13(#j/ph9:5B*ym5c\?B#[CS{^}(J4}jNAaO[bC0C@0Cm-4eYP81Qj]c1>&$M1)Dlr3i.vu2-nMS"
    ">>+R/[^kf{0Rt^^\?TR3B{.}J\?@SI]]!7<rB[!\?@GSRVud:PH@-].ode65<t(2<z%05DAjc{6gl!)H(Y8}mWQE<Qq{{@}t%\?}T/sL0<gz}]9+7b}uNfS\?Ty<O"
    ")\?-nm-Pn&C@6Z}})+^S)]7.!F3Jo[S}2\?z<28UNcjN<362[ln0hWoP[]:pl=>>sYP]9J[oFE8IQ2[W4i{5sB<0]KjH5axS79qoEg68qpP8v##R{B#QH6aT*z"
    "]z!BB}w5]k6:1Z55^Omp3h:/w>j!-ZV>9(s{v[$80@zV=bK&tk1{1Pg].sWt@qMA#3haz^>)vLg(*BfD(K@Q*[C8+x@RDD!0}#YN0Qy0Y4KD(c\?#9uF0Tlsj"
    "\?7*@]jqEI\?]57t-].[QR@Srqc:p)\?+2]6@<%lG$9@z&{&2P%xL841L5/Zm!<1[rrI0sf)D<}[Gy()0N%{#y#p/DP(5*Qd+e1p>X=.XrpN{vO{Y[alhR]-B)}"
    "]1>v]{5U!c5FU.:]=ab53>ivI7WheA4\?eoP:qK!76A)}-3HlKv!8QR\?3i*s(%io@@5dFpd5bM1<FJ:qu}#!FD3ChVOjIdr}2UuizWe*7o<0I1@e\?Muua+Qvb"
    "0jqLk4Ix($9s7f][b@sj)I>#)lM=@z}@$^y94Q[k{{]FC2nw0V5+DCy+^O2r].WbP0WceG=i-RH0l%Jx{5(ii10oC&]7Qh#@#7:Z)gH#:5xK4Y14N}odBeXo"
    "]#cs<%juME>NfAw*0)ee=!DXYfFrRGjKURH3I$OX)iZW71Q0<60-uPPj]UAZ2L^@(@SJ051pCCn0R#:6}UHf$011kH@@rD31Pxkp0%XM4(\?{kx4g-sl{Ypu$"
    "@pGX0%nqrU2&#l(1{Ceg0}(D*1n-5Y@qcP{2<}S<1OJ2*@%O-c@{xdQ1nI(@}Ugc(}s>c{1p1]g1P[Sd@rJiu}3vc5{Ypuv}U/<Z}2*S@@qb*h1{j(e0qUTS"
    "{wUf-0%Ou00%nC%1QvF-jL{tK>j!]f)DTfC[9y5r0%l7A}2XQ})f:c(]+n5q1on2{@@h@<}(YX<!D80q0QG%B1PP8u[AoA]]AjjV0oT1O1{BYa27V:@}#Vr/"
    "*X\?44:}q=c-PWm1}2Q:g0QyAN67>LP73=5j}tyr.{y30(6Z#bz{wB).6DcAI%0nr<(hp[[}$G:-5bDks1}Q$e4GMH-\?Y@0{{6QS{6e$Ssjo#jI@qk{q3afS>"
    ")eXNJ{#.+}@Qo4E00SJa@0r@K4eQ8e3KuLd}2*Y@2Mi%11[A/93<3Lh1)t{>0%/kE}XnEH0rSr0@oCde[D/iN4Eja%0sFL]}W9Ah@0iz[3\?aVh<N\?AU0pf^-"
    ")enK74!F=-[a35D8}>t$1)(<m{tubN3gmr)[b&N!}uWuL[7[uo3K%=J].XBy*$@2j8j$]ijq4k\?2LxUN{ZWeq@#gn80q(Zf6:<)c]8[sw2NhKD5BHw82K.#]"
    "^eT0J5eC0q]9k)d2LNJk)HZxa5a6$(boc<(9SulO(&b}N1)$*y<<J(e3J/XP4gz7b}#sUA[cwjR{2V8W><=7K2M:Sg(K<va{U9}x)=KY^3g[dE1P!u16y=Ip"
    "[cP(h3lp[#1[d(G%nre>]A&Ll1{K\?g[0ZkYjotDz1]!ey}T%ZW%mVlC(J07S@q-X$\?Vi@!)ed51@08*h}}7#P2kO8)3Joz!1({i=}U}ih2k5h#@r&-n[FHAR"
    "@Sh&[0d+g=2m721%nrk@0}L<\?\?t3<I@qV->1[&2M\?ULN%3&Zqm{2lU[}1a4V{4FdN0%eq/9Sb:z]AU3](H$qT1nzV&}tQ<z0%F.[4ef\?)^G6mx}}.8G0EQM9"
    "jO6cU3czJa5yfzd(j8zB}Zz4q}owED0r8W302C>h93>mN}W.(#}Wi-n@TxzE2LNY-!C]BR2lCwl@rHWJ[a=!L1Nk:!/8a$T7YjWB5E!<u}T-!U00zu\?2L!Al"
    "@%!965^nEl)GBaJ<QgX&5EOQ.@]rtP0mge{8rp8>{xo*=[AoYy[cw<.5CtwB5c%3{}%>r+%a7<.<)@n)b>&cqjO69t/C@AI0O=fbkFNE5)EHY*^kZ)91SVU3"
    "@{i$t3>&^j2)pRK1@()SeCe/sbj#=R5YsFP&T1D+0Om!$>i-YIbieK}&ry8A0%[}#1%t-M{8j!^2kFL#d:*c:FZaU]@TFzQ\?Vh{72ilyo<llyB^<<ex4<hX{"
    "&R%m>2MUOe\?M(>y&Po!k/ExDd8%EaHW/E6v}WqV:0n#8@7!O4mjOxS&6BaBU1Hdg*3<1V:[7d7v3jEiSbhim%{Zu)x>jXD:%m3U/90llR[ECP=8X-qE=>D^a"
    "]xWIp1Q<-4}@#pP[avpW)Gr.^*1uk](G!+@68zb4<)\?K03*nI&2il4g]B6yP3kiN\?)(-h\?0t$wj)iyQ=&2q<@!:q>>@pQI@@2>}05=xMQ0@&J%a[MlN}U}x3"
    "&t.$x<Ptk/7$=<zjNSKJ@ri02]-jN*3h$wu!D(/[)!M\?O81v.b}U{K22mpzp0r\?PO]A+V@{4!QA0TFb.}%aQ/@rA]K(F-6F@]i[L{sooo3jdNQ%nibV@}3u$"
    "1[.&[{W]45@P2d0F-Vn32]1GPYz\?{k0r9k#1O}(Z}$ZP\?@TxIp0@-i84d%P03Lraa}(fX}01n8^)ev4>2Nf(7%lZ3[0{4=f]RpA0jNJ6#{v5[r3HmS![!]Z:"
    "}PYSV(&w0F1havp]C4lb9rL<a)/R\?p@O4UJg6OAV1{CU0+ja%igW1@8]5Y&![Zc>f)*Yj5}6UP0<ok2zblcJ{1NBnl<[!fuFDLYR<p8v=%lxam[D5<+:>vWN"
    "0Vria2Lu4vbME!W]By^7.#zEK1n8v5>>+F^3J.4M0P1gl4DXiI4eQF7{w+yi7I2w!joKWp@RcM>3.^O0@T/wm]Akr#@%On/1POS8{3q>}0sXV53jD!3@0ru6"
    ")/9K>2lTY11Qj@h{WOy&@SJl5{w+wF{5UA$3hs-9@TF(o0%562<n5M:0nE\?H[CJB-0SSe[3i]\?T1Qt23}v08(7xTBx1]n33}2{Zg2PH>\?}WJq[}#=<a%nS32"
    "0s]zH{XUc.1P5Gh}$6Z=@pQ3\?6wR(fjP26U\?#ByL@RbmF}3d5-F:rsB}3tyw8$%^N3\?k886ajmo2]0{R}O&ve0SA612N}KB761^A>m[Ec}srzx[Agh4}t7d7"
    "]yJ80@Vy%#(hZ$T1p2hv}sLo\?1PG2h1lxRc1{A/n4flRI0#5=P}%TaR\?r:h8]y&bh>Oq\?/)EP#n@SiDl0}MG[2ikNy6+yMi5IgYtHysN-4f#L6]Pd\?kjpZ&Z"
    "}Wr#i{xQ9o8Wc0p@}M%9]AV031p:=c3IBU<5!$PtchD*k}Wy198uQcI2mpwd(NhwL0P:yF@QxpZ4f3tc1loN$(ilXH@QPpx2L)MP1qHwd0}kWY3hTh[3iZqd"
    "@S%2n{Z27H]zedV4FPAb@@7{H{-e@61P!^S2l0<269!p71p1LK}#VA=4Gk2+<]keN2N7Uj2(0#=2x1vfjvx.<3FzV^0@J922&(6<@{fR26wp$23C(M90pfdG"
    "2[3C77y6#\?841C9&q8Sy(JcOFR]ms87.#Jr5!R[R2N.<xF-cU3}W8//6ZC2^PW:CT@nDxj2N4@=+Ns*k\?#0JT}U*3O74Ze.^/*6e5wWIH(dj>f8UKBX73A4f"
    "6-YxT>>(.Z5BdjI>gvoj<nn&P@U=1N3<LC(&{lY<)nZt1j]kfN0ttW>0}ul}@%YUi0%n2}@%nu3WF*yd}2QE0@RVY]4gh$h0T]5e@@7yP0se#c@@i062it))"
    "00&G0@@i8>2lthi2kO.%0TwSh1nScb)F#.J:o8Iw1[\?AG0SJq<1{1C}1OI-%}1BT>5^!7j@}co]3<3td1S^pw0TnG4@R(%01Q1@g2(U7^1]ei32mk$V0$kd#"
    "0SSx33q&$Gi$6Ny&>Kp:<]#\?TQE:+:0O^RSh3K-w4aAl=gF6qD(hG(G}Y:x0Y+Jo:S$mO^@[n7)*wxr5}Y/>F)9C+1726Gx(GMW]FDr\?k<Jv6g4AewB4V/@J"
    "cgeUY38Dk@@79B(d/QvS6AubN0h}sD[wCFB7D[cR.t&M@{2N9^9Q2=Y=\?vh0&)lz(g8oyg[&5n/.M&.22[lE]9p$-{@][gUjOONR}#VA(}U*N2@%5=9@@9lc"
    ")d*IS1O.r:4HacB{XjT\?3&Yq{{YpfR1P5n}1]ew}@@i8^@}2mQ[CA4W^f6*.@}2:!\?#@RT@quA>3ITV1}3l^70tthj2k/ht4ho0J@r-D3@@Ai3{YZZk}2g0}"
    "]bD<1}2n>y5buiR2j}=U2MsI$@rT$D1m=M:4e{5i>(}7*]-ki{@}Vu!}@\?rwjP$=U@}=P<>k+O-%nJ1G]-t0{@0r5#[+@Q.[=wZZ]-C0>0QY<<1{1S8}Vc:a"
    "1Puaq1oeGb@@A2@1owDb}t{N0{x76.}26<<0ttFh{5U0-[bAs!2MUto0S8@%0@So3<]ktT%nq%21n8oX@R(S{1P52!}Vu{c4*8+m{6Q=[@@8@}0SR#%@@Je+"
    "0R#&@2nm7l]-1)%@}=Y]2+CH.jP2V/7YSjY<P13SZ7\?d-3GRH4]B\?o4@#\?1S1md06\?%&u5}#+*I[bq^#2}+@O1@#!o(b%j$}TI(b7T.KU1rdpA7w^{$0rI#("
    "0s:RB@Rc3Q2(bee0P$ae4dJV/4fMFg0RlGM0R&]91QtpM2g:\?d1nygp@2%R<]:yt:1PbXg*yabf]7{&n@R#c:@%EAK/8tyA0%X/a1QCLd[[t(Tj]1:-00H%N"
    "1{Uwm]yz}:3Hmo)1NM0*@@i2$02+Rl1>A!G[=!pX{YZV]>k*ej1[Y$p{w$9%{Xa<.@S8S(0rI{01o!e-4GDH@@qDJ!1QjS3@PTW]@P%v!2l1P61PWY-@{/</"
    "}tQ*92mf&b]y0a^0poBS]8FW>0%(KE(icez3Ian{]7SgP(I{}x1oeG00@Ab#0q#(&}VlP\?4-xrejquoX83CG[gZ:S:@#}sA@sfb8}n0Zz57\?xM0RD9=}@vw1"
    "<{7X!(&m[r/A.GP59i&>{WE2V6z&%D]8!^n}W%>C@sw=7{.JC5]wp8U3Gy:Y2mIT)0]J5t1>ShH0Tdi]0ToRH}uu{f[flEi3HMEq[-w]#)*n%F3KW1f1P$dX"
    ")E{V52OcK>)^o3A7ZQY}}t7uD6AOa45=\?1T{m&/6jPt3L3Mngf@[EwF0S-Ma@@rl51]mVI0$CXh@r@fl{u}LF1p24%0sw=40Sh=[%nI)23gOl469Xjx@RVK2"
    "0q+@h0@.=>@RWbX(\?HCj2mga<3&Qwi}W02](JdD(0@&t\?}18L+3ju[h}UyfY3J75h0S.JQ0%XX$@}D9(].F%J01e(r1{$!t]X*L@}0d!U1PdS.(Hj=I5b(>q"
    "2WB&VjOFBUF-C#34/N1o\?q!+n3[8e.5b^s:]9#oP(kd.J7Y%Kn4H9aD@s/8g}U{:4cfVfF@V7v]0TX-403}6&[-OT684d}S\?Vr^y3h$zdYwL%L5#Upj0qU1M"
    ".UB3w*Y55>@i)0!^GG<w{$Uo9!FuQ60r&PN^ub4}!=DTe(>y(WdbhYD:ijGb>k+O..XJjH49[+!10M[5^H(@W{(cUNjok(Z2mpvy3>04g@0j86XAP.3@r]h3"
    "}$Hx]N^f>S@%O=6)^{C\?0}Vx(@q+%I1m#MP}Ugi$4eG#v3K(}v}W9H#{vwH$2z1919qfjg@@@VG0SA37Stbs61oF\?%4ex@20UjO/1]!Jf}0e4N=Jj)+3Jp20"
    "2myBK1q7hc{z@&d}2pgj1O\?%2)+!cF4ex!%4d}^NLL+ZO1PFY44GkJrjPCPt0sn>@2LF2[}#bNG0Tnr5}VlfX(&mg\?\?T{%R1oFJa0q(/}@RVM[0Sz)20rr8]"
    "0@@=i@r-Gd0rSf4}VcLB@S9i7@]A[U2myFi}tpv4@@Sl7{w>u{0r-F@1]F=0{Ynbi2MaSb@SI@}1n-960@z/>{x7p31QB2)%l/%/}$Zo]{5Uo\?0rh)51o5t{"
    "3i/F)0r@A11PeD924<y+jN-y.]z[dR}#WSh0#6(01o[M9@{*i\?2ljt<1o5=a}VDu-}u=]@]-+/\?1{L25H6M/w{jr8B}sU9Y>O9N)%nAxa@PA}\?9S2\?>}$gg2"
    "5bMUv2l1qO00@-9@@hf-}t*A[6Ba*J{6p)a1q71${wK)]%mk{.}27f)@R(<4@SzuV5+MFi3JQ+p4\?P-M@}#{a}Upc(0%Fl90%4]/0j/X+jPbp62M>e41R*}l"
    "%nIV@@%wVh1n}Y!1nSc1Oeqbd@R#^c}#V{7@%nM5@@zJ{2)g8l@Sr330t2ee0rh{4}#V&(0SADf0}ux$2j\?-{2L[F]4eZ2dFDCPF0%X<U0S-Jl3i8pP{y499"
    "1n&P73<3eg2(T=5@SJk:6-4+j}U{Z0%nr0b00Al31oPhf@Si601O6}t2l1G50SiDe2WM]jjO[*R1Pb>l0}=o<<sQ=n@q^GD&#ndv2(<R%1POz]75w45}$H9:"
    "3Isn\?1{1=b00ix90SA9o1Q$he@qMD*}ska4]-2i>}3MJn0%4PJ[Cy#E}2pWa1]!A21P!2@3Ku$o4GVOt@R(&=>j!rR2NQ1%{x*3W0R#r\?1mMoR[C4Og%j)hL"
    "1p%Cd)^/Vw5=r18]-Upa@@8T3{vF4K(PdDpj]b3W0%3*L1QaU!6-NX91{+8n{5+x.00qV/2lC290%wC<@rqS*}1<j!@OmhF3&y5b{2tPv{vD=c@Qy<276jEv"
    "%jEtJ]8XyR}s>u$3jcFg0%5Dh1o^/<3<2!#0%v(>@Sr5@}tGjJ0%!Yr@TX(g@{]:C%nxXk2lsvD1[\?0M3hBz/0rHjp[AHnU3Jvdb)evj^0S\?#)0%nhM0wjAz"
    "jobx.@@0970@@rh00ro51Pnk#[cG6>0%j5w00rxc@Siu4(\?{>+{YopB0}Dc!3h%/6@T!]E@@qfY01Pb65^eyv@qlx&]A1{L@r8i-0}Mx{1nI-@00icc1nh#$"
    "4e78l3h%#f[+.>Z@%5Ad@qk^Q}ZqK.1on=m1o[}0@@h($1{1M5}2HcQ3>S<B@}Kne-@D5s0SAu72k53a2*}mTjPC9F@pV}L1)L):72oY3@$:(d5=I$N@q^cb"
    "(Lf)53hiAI)gF!d0PsyE)e)vB>hbXs9piH)@RkPY\?rjzj[C<yh(j:p069GGv2nmHY3MxlN0V&P#2)pqi4F6Vl66[(XbiF}o6w&&.@2kXv*Y@xg@n#.\?\?23Q:"
    "7#C=m2n+t>7Wq}K9ov4Q>><i:3j2$-@@@J5:>U}y@rhY$][cC(jO54C@pxQ!)/<p0}v&&f3!a[W2(1+Q@QZcE2l$j$3K)Ej{xn!b(Jl$v]zwt0)eWs]{X>Bd"
    "{Vq$G3G7%}7Y<%x1L%jR{5+5v}2/<O}uDoU0sF(c1O{*!1)Vl$2mZUe<]UR\?>lg[!0$L1[{V.}.0p[It3h:[}}rN\?M@p{iB5918.85hxu0O^wm3&Znh0Srn="
    "1oO@w}$QMV*4FksjO]i[0r&q>:+oD)}#E8>&0G5X@SJ5[2kXS9>>LnU@09i82l1V20%OP60q(&{1ocBC0S@Me]xuqI@@&u6]-C5x}4Ao91l]H)0Sq@[@@S9i"
    "+MF1x>k>[!0S&nE0S8=:4G(d$3<)4s{5JXv2(+bc3Jf(5]Atf{@R(A]@vOG^{XaN*0r-G600jq41Pnt)2mgC9@%wL(1Q1V42/Idbjo$]<.V6Pu!=u]^<{]xg"
    "@qWiv66]Jc2j-8!5!Vte\?5u[23)9B!/=p!w2k[ln&udH>1<NhV@{fg[@}um7@lKWwFZ\?SL5Di%n[aMX-0TgWY)wG]>4Jjps1Q%}#1[o!q<nGh[4h]015dm*&"
    "0TH+l3Kc\?Y}3m2=@REGr5+hpX59RKC0O(ry]yjTw+N+fX2&#J]{xYBL8T4Xh)=:FtjN0KD>/jkC0]0zO00q@.}$g0&1{0DF4GD+C5ebcU}$Hyc0r9lu}40Ao"
    "]8wU6].!vRS}c-9\?$[l8@r8<5}TAEt2I]a^]-TZ\?[dt0T}%:J$<[5rm0sOn[}v&\?@@}E630t2U83KcIb>m!V}4!//l<]+k&1%:0rZC26^}thb[4FZ86@}#K4"
    "2ltzL2)QUn2<\?S!F=t]P{5$i!9-]V7jp8\?%@PT^L7sq*J0r-&6{x.6g1{T=1@S932@$UwF@0rrk4/cvM4g9ax5Di1s}uE5*<P2$)&roF/0TP8l]yRl$}3v3b"
    "{wjE^0%X=3@@&kRS09j$01O]11{2na@qN93{YG*K5+Vtc{ymc35dF*N0SSJg4gh$m@ortF@@Jk<}tfTW}$*i\?0r@YG1oO=h}2/yL&}&=L4/2w*4xJtGj]CuN"
    "0%[!<0q#(#}#m5L1pt8k0SzW1\?tEDO(\?gtD0rils1]!X[@s5=93&PD91nIV{76jHp}W0r9}2*W4}Wiz}@@-(q@py0X1}7nd1PXVh{ws{%FDl2Q@%wU}1n&w<"
    "]8FQ=2>[D7{YH^51nq*0}1::<3&Q2c(IHF^%mVx]]8nH*@@r91@S&D80S&G51p&VH}1jvZ4fcXl2f{b8jPa%V2M.gh>lp!X}5Pnv4EB9j2djpp1oX.R3h1Pd"
    "{S-<r{6w+gZ5OVs@rqS:5En*H}n-mj)^\?dY>/9.$00r637X4[w:PInh[9p>-{zJD7}%C482iU%Q79&tK9VKbv]wP]J}2H=Z5c}%@3hEB82\?^w{(H$b-6yLD9"
    "[A2if1L<N]}#4>-^/yDY3loh-<MV<D3<m4A1{K$H:/NCOjo+P^1[}-@0rrD8[EaW.0OEeL>&8V>1RF6H2M$Cd}U}98(Ju\?Y/z-Ry1RyOq4E:D00q#Za0s5]d"
    "3IJuJ2N*}x0S\?@@@q#-t{XsT&2jJc30TP881N+:k}uDG(5A9=%@}uQ0[Cv!o0}33:}$H6\?0%w0p@16XA0q==+@RMA-)fJNR}tIU#2nE*K)/97S(JNm>@q3Y-"
    "4G=.v]zJ/4jr8os+(k=K(JubG}4a>Z)f0UP]!cP)a1E{d67E{%0Tdl-dGxRS4\?v>n8@8B@5F#2d5-.^VaUs3a0>[4j}oHGm\?O52&0$>e2{SKuF59T5>0hRjk"
    "%m2vL]vJ<6\?246k><m9x@rJbU^!hR[3>h8icd@3G>D:$n>OK1a1)$*T/-p%=ZNTDK1hMh=FDlIK0OwIf4&+&#8rZC#0\?a=#jPUxW1]Fb%}t*{0@Tw]f1oele"
    "1)(o\?@%eU}1Qtwd}TT<^{YH0^>(7UR0rr32(IHU-1Cc7C0q==[{uI\?T01xqr0Ubhe@qVAM0Si951m(V}0SI]=1Qb553h-cL3<MFc@}=Y$[=n**2M1b{1n@oa"
    "1)L*:}Vu%50%5rf}tf1L{5+*c}t}e%2kFo90r-k&%nJh)1{1R%2k[/00f@E8jq]ag0]JOT5B]C&%0exT4*r1x5F2L883P9#1[9o54*}$4(IQIP\?s.Gt10vjl"
    "O8L<${o:V1]w7wYVHF8574zUp3*w&s.V6Iq>Nu#w8Ukup6./I@0TdJU2N]z^1Qb5q(>*ig1o]tB90Ezk}UYaX@nN966x]-t}SGyk8q:PM)bWWb01wGqtoTB="
    "3<)gw]y8T>3h9JY3[}Fq[n#xAjOFQ+@$a@J:PTB9b<u9/7z2Kn}nry0].dt0{6GZ<9s@FO<]>$/{u}ns!byf-&l@hnbn6U=0stQN[w<q$2\?5XLXFM0J1p(sF"
    "*W2Wo4BJL*1T:f3]C=o:2nlX07CKRa>h3yq%j[{Bdgk\?eajf0W{4i\?6@rSiJ{6.-k{rSm^^^(04&12OD&sE*40PqX@82<aV5FaG!6:b9F\?(3X.j]j{B}1j:S"
    "&rGF:}2g!P{xZ{a1n\?Y>)ema=VIsCj0rq%32>Xo41Q+8a1oeJ91m)316:+0z]x#\?X@SAi7}tp3/@qc^8{Yf:F]CWSe@q(-F}tyo(0%O]b1n-Ah2k!!)@%eY9"
    "1nq([)dgzN1OhA#@RM/@3i6(i1q7Rt}N3m%}$*l[@Sq#}@S931}2HT01P!A2{4n4(@PKv!3k>UAjOX*V1]5Di1{TIO@R#P}1nz)g@PsC60s^rN1PwC}[ceWT"
    "1oW]\?1o5Db7YB7j)G9g!{RDKX}rWq8{4xp16-[Qz66ol1}tYWa]8N$U1qp.l1Pew@1o[t<4F*nj}#Mx(2M<&5%nQ)Q1pB=h]:H<5{2t5@@RT\?g008!B5b%-S"
    "2i=D>4fqKU3*u*F5cOE4\?#B8P2(<u43&G>F27gn\?jJD1IZxTrj5E!743w3Oe>AL/.>(Hh&]9t/!@ugv*(lVug@i+Uc0SAh^<]sGv<&uuL><5(I}sbfV0p.rH"
    ">(0vg{yFLf{w=fI@rSeV(f1#}+K%JP0i3pX}34So2kOza5^NN(1jW\?+\?2FX=-oGU!<mpjM1]wb1@2A9L{wL]T@$3j[5yqu$}}xj2cFwXa8t0w2@37.n3*Oi4"
    "gxArljP2M=@0@L<].ev!}v-mF[Cq!^@Tv/.@%w#gM^#lv}uM^80lto5{1xfx1]OJk@}V9H68AgA}rn7<1QsS1(IHK[2N6e}]QEco@v!0F6AFj<)eWp%0R(($"
    "1PX\?}2M1tZ8U&d.2(>8o01Oz/0@J6b%nR-#=<G<J3>Iwa%1:ME)!lX!{5a^X[alFP1oFX}4i2JK&SO-q2)oz!5:\?P^jpR8\?(jrjQ0sX@o0#7kc0#Q>A0TeRT"
    "[=X{%1]mcu@q=&@}%<aN6B+GC0%Fxb9r39h%lYW.{4ey=@0Kwh(jiy![XU7%4H8tN}tgl{(45*u@{gj54E9O\?1Q0(=[b@KZ3jc!i00&xe@}l<p}%BKS%nJAl"
    "2LW/>9Tq&K[CB6{0})fh4fuv:!*.7f2lUbr]-s{r0SS8=@r9Ab1f9qmjPkAQ7AH3+8W)5E@)R^[Yw@C<]w!VOFDb-w3mB^F0TGbE3(6rz2*%g-4fkrt0Tp9c"
    "82Tjq[B3MwaO(6G2p63s{z8ZD}}I(:2]ia1<OD$8!.{J6gr*cz0TfkD5:f-t6WuHD<L6[p^&+[0}%%$23GH-SPW<nY1[Z%<.2-A>[Zv4B&]]Zu\?V.Zq[aWj:"
    "6xP.#:m/VR4DVZ^4yTDxjpzt#(>5d!bmj[\?a@ql<[Cr+y2j8D(01]t2:o8II8r.ka)*G3V3IsI(5^5QJ[5>j>4f(wn6-3+P\?u<bX@2=mo0@-Y2FEhhV@tDH="
    "4gT9x0n5y)>>+[}@QPvB]a7B0*Uz/90k[)z3bvRUVHW>Y&S]X<5Cu%R@o+:hcku^>%leqm{TP@X<>o>l}1:gj{z9e(3jVF4].5v<(uS+Ljp8w-)id@b9%pW\?"
    "+&/ci{sPxC49[$w3HuKC0o:jL@quo^f:50i4h6xX0n^zF@#.yt&%VMK1OJ3e9Qr*^dfnG7\?rTN>5c00h3)U(<5:Eja4MTL50pGak)H*ip@1Yhe2MC>3{XTQ>"
    "^^vYn@Nij{{a=*l)GI[>}wFmU>Kv^4%04:O82LrI(6qh&4*I.i=GjTh)^x@C]-TpW{.^A!jJE:J)C0Nf3jONefa]3d3KcXR4dTS46m*qq)+]oa@SNly{Y{g8"
    "0m7(hdgb\?E{xg($\?TfNCebOX}9rWIQ3L@)q%l}9E^*l9s7.Ex69%sB$]dGR*>&7Fr8}KfT&VnEk@%/Hu2(#m}1@crh}r^Un*9Mp}6YW77Zw]\?Z1%a=Oi1vLC"
    "0}A>#>gQdpexB}f}s0EL9Qy].-n7m#5mNS3jN@sH0qCmQ[aVIU}56Xv2[cFn2Jtyy5deZE2NZ>u@rrM7}48lX\?%gap@@-=y1[RS/2/DDD@oTsJ)DBiv1OAPm"
    "7uM8i(&u*i69OWv8VPTr1N#=>%mVY}61^Vs84Cr$1o5Y8]8/0La]}Dq4^--P0R1BZ0rh*j}3)9b@}L4H[-z1z3Ir=Q3!uQd1q}ml4eo.+6zqI890bx&(*X}>"
    "jP1Tz4LvFQ}}.5E@0Bbg{z\?V[[Dm[P\?1.iU{TorI0TXPe%iy\?B@$twn[be4R2k4r&)fiB@]Z-7>3Ku!w)cKh@}V^f93*!8\?7.uAp4fc.R!bPEUFE8LY015#i"
    "2LnbY@QVrJ5b=Ru{YG%+)/-pN0R2TS3\?<xb{9vG*}4i2/{4XW<0p*i&2Ou\?E2ls@15!%%j&pVZF]x(eB6TkEFjoj%J3Kvmu}tZZ1{5TmF{XT{[@Q]<HF+YGN"
    "2(K62}s%)Y{Zmrg@p6N=3h:2<@$lcL3iyOt@}=V#1{Cei{6Z:2].w%]3IT/c[aD!D0qDT7]z]0/@%Pnt1{>zj3Jyhf@rAq$1Pwxa&rwDR}u3{e3IKO$[auU/"
    "@SqG{]zF:&}V^fd@S-Do}$HW04djz}2lB&f)gfl]@t9lG40.SajOn(#(/59n7vHnW2[NWG}VDmd0Uj+]:QOTH7z3%f]9>GH[D]V=5buL74g0jy}XY)N<Ss)t"
    "(:)0m8Ujgx&[&F%9T{r#67$Gf6c3&l@RWlnY-4va9T&hA]-aW\?2j}(^&%{!c3hBYk)!U=z@r03J]7iya1{1/q):cH@]yh8q[bh!x1kJ[-68I+b@[!aV1Oo>I"
    "1PEVZ@skE%3)f%Kjpg]P0$1G35DS{i3f.9q8pF=$2l1.>[b-)1@w)WEnfFnHF+OS^.VwY}0pZdp7vRs]n3RZ)b<#3E><e0L)cL!R7a{sR<o+8m[D6<t@@R<9"
    "{2(qh]zOg!2Dk=q@VI<t]Yue)4f[usjmAmv&OtmK[A]@E@tC5j}x=p#1r:lV=kulZ1Q<5&+N=rc93i}}5A0x&}X61)b/{q+5q0IBjrKnmc&$4<@T4r:>sa+d"
    "+H)r**:sJ6{MDcAnbSeY4FM^]=E&D%bky*x3H5(v]FtsO\?3AiN8XAe[4=^A@*a8.9MJWM={y!o)fXcO)2=){><!GM:1OqSmSRM&8(d(KC&Q=DA&{c.4>gw+}"
    "[DYs7\?u12B0ik]j2J3jl=FQ}DmhcZX0o0X.@j%01/-hMP57qo=[!83fL(HpB\?VGifjH#B5/.Y^])eFaf3jC}:4HIB#@vE!R5Cb}\?1R8se8#6Nc)f9Q{a{E-3"
    "[Gd[)Qymk5@p7sp>)dg$^!Szu]Ux5O%lGEY3jvB{{w:+61]68b4F)<y1O\?&>Y7b1>>iR{C[YRBA]85.e9$wT$%mjzw2*.dM3KEat2[m7D]7UC!@2t8Z)!O1E"
    "iaABg(+J@p0PTMZ{.Ax8!=Vsd2WACmjppP.3JY+%@q(MP}#V{61{B(h%lP{\?0S./t0%e/b{waTR2lKO)0r}]}}@EUY1]eu2}sTs^@{{iW59@JA1QB@g1Oz-{"
    "(&V[M}#j[C[EK<!2M$Ol0@g%E3Job/@%m@}0r9lr[bFBx01wP9}%B)%@Q5+p{xZG<3&fQQ5C:uJ0}DJ>1]:$){uIUO1Q6zI0rh#}2(L1$68.E\?@FLAzjqde@"
    "6-^s{0U>)}<oa#Z3&o&l>gZW*0p[7y1PNDH[cw:Q]9+@\?{ycG$0S8P@4GkPc%nR@t2IfyB}Ug(@0@S/s)EY(I/7!}y{vNeq2l%L)01/$w}r).^9{zIU2&$)R"
    "&1Mkl{w@a[0$+hj01en#1]={D3fn@a(j<=(80]#L7ut(i6+(kF1KEw-6atiX{YgD]3<V))1ovyi{HC]9jPk]]&pvrZ[3-@d{-GRh69vUe2M<n%2IPaM1R7ni"
    "]ACE1)f&%n0rh#(3i*k3+KU/x{6yQ6!BA+N3\?BF@@O)*c}r/7274}}r1]WVY0q#JB)d!#[2lK(g1PFIE1]G2l].[f#0{*H6{U1/&1n9rG0SI/*@{GW&]40*t"
    ")*4OP(HalC1ono50rSSe1QUIC1[&k)>kB#+{x*-\?11kgCjpqn8[-{+.3Hdx)0V}G!{wB1I1nGkp[brd!0#zjD)f9Ln\?4w=D3i6R-3JOSI)GVRk4xQry>F4(z"
    "*xdcT1Rf}%(/XoJ1l*At}$PZ=0}VK8>N!D&Zxauj3L@*D1][/y3Ku[i1oYFl1NeL=6-/9=\?V0xD2(scz/.%Lf@]\?!#7YKP>0S&S73g[h>2ls=e1oNSrFFd${"
    "5BYY2b.hu(jQg=:4fk>%@S02/4/#Rd1n8M{@($Ps@@IP/2kXG3{ws^+%nR@&}2vP82aLU}0q(]%1O.Hc}$/{W2>m&^2l>wo2jhJ-(Jc\?+}V3l=@08##1RR1p"
    "1P)VV1{BL#1]W(>@Sila}U(6w@r8{c0R#=$00z=S}u=P!1onL\?@}#V]0rh%3}2xZZ0}V=$1omiW3Jyb71n&J22lK$/}$Q(hjp7&}{Z3D)0UtIn4fM}85FsWr"
    "1{j&b)fA^:Gba!11p2qF].v+B{wa^!0S}G!@Sq6S{Ut=x5DS1u0@R(/}#v8:0Ru*3blNDk&}JSR0%FF\?8u6)cST5I61mb#r3jc+d!^zJZ1@4aw1o/824Fpnf"
    "2kv/)0S.}@@$C!z1Kloh]AblO4fW7o0Tonl{ZM(]0{oKP3&6+@2f)OP3/iJ:jPLYR@S@X&@Jk+)}V3V<{xxpP}VcS#}#l<0]6VRQ{Yps50RlD<0%O-a2(+1%"
    "1oF8(1Qbbj]9tv42M$2a}2Qhu0sOP5}s%{<2l+2a1n&ib^/PZR>My$/0@&F#1O@b}}2o^-2laVd@q^ca0@-2#2k!n{3JH88[BVOZ}#^93}1$f#0S-l21]539"
    "1n&x31{\?Eh0q=S[2laG2}w=O]jLoe#0qc3W90bZl4/#Ik0}uEcb>}A2@S@]m3Em(F(<1Nq}3NO*}%TdSZ6:n[2KZ7DZ6TX#@0a2n0$URd2O3h3\?p@L8+kh)7"
    "]+c#*9$k)<0]R#n4DVNp@qu<k@ShV)=*y>]>)4qe%lxy&4b.p65C<La3C#>E1P!I]]6VhM]56i7aqCs/6xHha1muf/}tyJ}5zWJ)}uE#n}+WQ1jO5pI1nh3<"
    "6-xG>(*<@G@sfns3)CAz0gjrk4G)sr[b8XP@@&F)0@qY#2>[n=)\?l611kJh74c4iK4=uA{>Pw%^F+w@%)/-+I}2Zo([1]vD3>:fU4f^V#[x7UA>(}7@1p1-0"
    "].e7^&QDQN@{Y:z(kphU1[ZQM*wH:K}S!pJ6ybAk75+@C5C=Rg{5tp1%nri2<{ZwY1rlj@3U-L%jO/.x}viCB@Rk{X{wU*m[^jy^0qVG]&R<bZ}27m1[DnT*"
    "@@06c}q}LM2(B&e2NH{K3jc$z%gOQs0TFL{@R=Au@rrVj{$20y2nN1l)GJNX>lRg)}2{($1]X!Q2L[=2>MptD3JQws@s4=\?)F(!v1[}->}#(@))F>#E@@-L{"
    "]8FQ*2MsC$5bDFE4eZwi4g]iq)/iEJ0S.cL@3B!cjPt=]4E9#]@ONzN}26N:00Sl3}Vvl31OJ8$2k]FG3>rvg2lL5i{ym0583HlK}Vdfg{xy:50rr652(:[]"
    "{t>{#5C=tg>L:]O}2*&{2k/a%4DW#}3iY!]1[ia]5bC+(1QA-X@RlG^@pZm0@R=*e2ljJb{8:8e2nNaA@qMT3[9{hl}Vmfe@RvAB4Gk$@4EicU4Glw72iL<:"
    "7/4tPjo$-])=Ue=00<\?C4gimG0%5F]9(4P6@r-5D1{lyI0]KmM/-p:Q^E^m+3<D7$82+c\?2fbmg2ikNH)c:k)@tb\?w4-FzI\?2l!8}$.00{0h{)1MI6L3L03>"
    "<n5fX(>XZ<&>t^h[^D(t74yqI}TLBq%h0W&[dL-/&ry2z)EpqM)H*6x2LnAc[\?}DI[BCr7{uR+z3<Mh80r:bh%8>:-jPLGU}#^ob2L!w#@SI]\?}#u*1@RcA>"
    "0Sr5{FER\?.0rh@#}$pg1{XaK^@}#]#3&o(b0%eh@0TX]2@@A2#@r9l10}V*4{X$G[1oFD6@SJGc3Ijr5@R=V{@S9f70rSn#1oePb}1>9&}Vd6a1POJ3}2g6>"
    "2MsX@5=\?!k0}=D#0@&u3}UZr<2(+wg@@J932L[#g0TwMa}s>c]2hI6sjPK<p{.@#x0{oEZ{6pS-)fJv&}t*uV(h\?a/]7Sp]0q#)c@P:yO0%!&96-EC01mlH3"
    "^FT7C[cx741{sPd}33AW0rArj}#c/(0s5Gk1POPh)E/cnFDM2O0rSP71[\?#/2lg\?a2Ms/9]yqwR%m(V$}Up6=@@&l9{5KaX{xgW1}u3]@0r0320R=M)%nAf8"
    "1[Zah[auUZ@}V/#3RFv*joC@={Zv004G=Rf@%oL2@Qp0>00J.*SSihc0QGTP}ssmZ}tH^1@rh&]{4(Jo}Vl0Q3JQtp@r&Dk2lUCu}tx:W]-Lc^{X>v0@%X@7"
    "0qDf(}3mlhF+o9o2kv#}2kW&/[b&z90r@J6[dCD[}u(=>1o5J23>rBW[dl0j@%oka4/uzF}s:TY2mQp*0rrb:3hr]}[+R$J1R*}x2qA#=jOx3O6z*&b01].m"
    "O+7>M}w/.E2>WY+2m*RG2K\?VT{sGJy4/Md/)!v502MU270%[C+0%m90{7a$q2NyXl]AB}66aUc@-StMm{6n\?b)fi<)@qDfY@Sab5)G&ZO3i6=5[-ZqN>lg5O"
    "<[nrb}U7:b}7@=g}#UBV1o5h+0pZN9X=\?/@ZwVNH(K9N/@qTm+FDt&S%nh/f3&72D[g9adjML9d@]K1.2{63/@Mjm9bQiQi0{yoW5!%A!5:[@R2(j>*}ut<Z"
    "<Q7k0*yTU]6aB^n*[1iT1K[Z&=^Tfs<[FlP67bz}10C@sZ1@.)2>)rJ}}g#Y2iCNX8rG/S5!29y)GBf:]:AtX[DdFm@]T\?O]%&5j}0N[&>F=:a&phv3/wl+i"
    "2j<tx(F7Ld0r92]}s+9O3Km7L{VR+5}PWk]jrKk49T7o@2P1]X[d2)v6zqa&>A(mV{wUix1{C56:<\?^k3HWD-*w}fY@SR%8bM[5J2jx2g]A+AX1PvG:&Uh[j"
    "0myR)}#C}n@mrawYzG\?R6x6b$0#g<ia%(dt0r.D{0o1j(0@z%32O^)<76#Y92kO38\?$ED^df6[{!ANR*)HF<$&%DlDQ:coj}V3xQ3fZ%=@quJ^aQXhR(kx5x"
    "jLY&n0[Vl02^0U0<Pl!m%k0!S}tG$5}t}cd02L[20q3xJ{$=kO}tZH6{Y}A54bHTI2M1q#-UevF0%5c65E5D7{Yo{L2[Wp33\?iT80}=}\?<o=4Z]92vc[Bc{s"
    "8q#sN9%{Ke1mV0T)J@27>k+!X{W[770#pCs5B*57)f&j>3(n48@T^DD5AixS@rrGe5Boq**.y1^)fJ)/93y>Ejt(Ho[+Cl:{YPWI2L4S#4f#k7}V3$/@p}x5"
    "4HRp)bQ1w(c<{mX1%iO6{uIng}xk4{9!AOC3<lh7+}x8g\?obKg4&#O>!DVu&cH]$t01Zdw*%b=*an>vyYqtAhZ1}Af]ZIvX@UUCt6w9M*4bxIW0tV445/qX&"
    "Z$n2[]9cI>67&Bd!X@aQf1zFE.S\?9uai1fB}xVBa.Sy*]@pS{PjpQ/&]+0t$[DxTc&>+0:0RcP$}Tk330}t(#5Bvr$}vsUL4dAPE2NIvO^fo{/(=71C[C%u("
    "%m^2+\?%z6>]ZA4\?1kJUK2[Wgm/Y950-oBb!@]+io1mLT}}5xXy}S4$!2&cyh{yv]7]8PlJ&07.y}#t{t1H!Hl{45N<)dZ2M}T9LY5CLUtFIDS.[d2^e2LWSR"
    "2mHXi4G)vk}Nm&0jNr6G4+paA5tOZ:)H{.[\?vP(j)*FO3]6D+\?=\?/A(2N*w6@uz1l2KIiY&SPU@0Q0qQ@08%E2LmV#2(T#t]b3o^[+sB31{28x@1FS5(H+CF"
    "<OnDF1GRv#&%k5{3*XXo}04$l5wFqR&2Iya3Ij62}@>&V5xK+W]yrO4<N-DE0]r+Z{5jW<0[dbV5D9vu86PpL=vD2p>mdKZ3^1!)jMLAr{5&ho{x-2n^^$P/"
    "4y2{z&$h:M6xF@K3J%ulj1ISI}tX.G9pAT++{Qv55.k+y&}SJ-}Vc=R74&HP(CHNJcDk{2cLUJ$(jjA!dES}#2\?7y:12c]m:n^vP{1OZs)#[<+=*&HT]CMSR"
    "]W(Rt1H/q23i{g[iTV9BR2wVU/$wcP5ZI<I9)>F}}umJw{T]9ldcHanNb5\?f/w37:jp7/H3Lz>8/yoyq@TGwB@}lT*2<z^3[7m*e*YFC%1)#lr]x#nv1RgFD"
    "1O8a5(&tx9<Lgau{5rwK]CN/v@m\?b3<pRX3STadR2ib{\?>g*j5]7H@B>iS997rtZX@0:kh)GheC(C-]H<MkE$)=jAU@0h(E}]W%)}1BL7@tbp$[FR-E%0lth"
    "&R@Mw829W)008-P@R#oT}Wh(y<Tb3AjotoJ}Xn]8]A:%:}tz8c{Y{D{%mVoRFD=8R2[EK\?%nRS:]8Y9$@S95#}#D6{}s&}d{ZEMp[C8[*{xQr>}Wr8@@TnD4"
    "0SJeX1{$hc0r&Ph]8wp!0R=u&2>[IU3iywm1m(J!0@&l60Vfz!2lC5l}UoQZ1PFf92o9yw]YVtx{X>u<2k[Y7@@Jw[}#u:32mp>h(Ju$)]-$f&2$!)rjo(:8"
    "@TnU}9(v{S@}lPV00R({}VVJ^\?#9-X/7xvE}#DJ%@R(]=0rq@}1Pn062>vA.69FHu}#D<7}V()5}VD/d@Srxh]Y)dY4gSgn1onPV*YElTF=ueU1O-rp2MaL*"
    "0@OzX3f\?J(}uDx]}UoZ:]z4\?-2m7t7{-]kn0W4BG}ty6@}UHr]0r@Y00@Sb}4Hffn2>dS(4f)gv9w[JIjOvIn{8KIS4GvgT@T]N)>Mye/@1Pa%]7zzN4aaT%"
    "^anvC1mdAkeHHwYi$HxV1OF\?:ddN=B()qYq6z.sZ(MlDa(KG&/}Tru[stG<K}$7si]6lP0[^kA{@PBdz3<[0d4E9lQ851qS+!TIMODe6g/XU=Z0QXI#<)e9E"
    "@TWP+<U)td@QYdS}@=[w}tyKd20Ie>+nY%T5*cJLQ#QlLjohlj([etf5^4X2@Sr656:0wm(>f9.FDk]P[+rW^@p]KQ5a]@y5axL#ObeiI3h99A>Mzj[\?w3B]"
    "2NHeq]u=\?1]:{Y]9Su3M01n+5\?UNP#>IM{N&RBb}@oT^{2Kie>8@[*d[.so.4G&Y5)FuRQ(Gwc-MfeI.}2=u^\?y^TM2>vi(%l/{.^*Dux@qE0u5cApA>M72)"
    "efT3Zbjo#.jpg&-2(96U1nT2z@%/eb1[-9j2mQFU@Lx3u5c0KieD<z#8TD!.3&Zq5{X$x!}4Aub0VH!41oX]f5e>#E)Ibf}F=t/j00i5E2Ld)4}>jn/2KQTU"
    "2PAGY{YG*{4fVF90tteb]Zh>S@NyVD1[@t$@SJ8r{xoEX2kXGk{vYa72I64{6z@yk[6z%V4*&3^2KITc@rI3X[dtP>{6r^zjPs:i2KhGw5+kU*{y26M}uM0/"
    "0R8fxXDA3sFZ$FA%m#-@}Tr}W&1k.Z>k@)x2(+N^%lWD7]Uh7m]aH-[[!{YM&$sc-{1/Sz5*EOF!cMcz[=Y3>>&r0H02Lz63&QXWTQL^x[!zom@0h*22KSc8"
    "0}(0<5!1&{7z4k\?]+CBM0%6bi3<V*P%m)6x}UQD$2l1]f1L08C]r)9G7V^PljOnKV1[i04&S/ODZ5E<i2Ny21}}IP31qZ813KWZC0S\?-Y6a9^H%nBw$}${3)"
    "NF=:<[w1*J1]C^X(IZeCNGAJ>4D4y6<nPF6@q-5-1{COs3)%Gn1pj&e4aCN<@[DYe}Y2[R&%9PQ}3=-&{08C0@[^j@3h0Mu)Efs#8Us9}}%$6D0rXqZ54rOW"
    "01bLh&pc@@2(SiK3Jf\?Y3&:7Sjp.fT<q!^i(\?gO>@rTbf[bI!W}3<N.=\?mCm}56Fy@qU^(2>X.c4&>6d>K!fz}4ALX&pW^0@SAe]{X%Z)]CEtS@sfes4dtLo"
    "1pL1})f0!{^^(!@(idvV[a)E>1n&VjcK4&W79A+Y@2Lho7Y:pJ}$ZA/)*GyB]6%W83(GeN[=5p:}sC3@2)pF8(*s=i3&tF++[oQU<{g+[19IiLjOO*P@q=($"
    "0rq/#^/Y*P0%wJgX+(Bh@q}JI0%en}}Vu@{]-1EP2lk2010vKw0@-5%@T5JD%m(S&@@&8{1]!Y8%k%E!1owA2@R(&$2l:(a1m3r$1oFA21(oE\?1P5e$}$Zc@"
    "}tIcb1PFC]@}#(#@$B@g@S&r50@-1}@R(M(0S8%71]({G2Ma]g%nASd:O=hz1]X&g2la.}16%4qjnF:S5b#a[c)PME4m&qKTf<2H5U]uqeux!Ori[H(f0B(T"
    "ZWrCg.T!x34dM0$*svp+(OYdK*XAtH=G<u@<{292FRHYu&0{K^<PtBR@*JSW*$6*S5bn@M>&AJ(<q=7[(OFB^58FrUQw@qEa0[@4)y.qGOvW/X6A[kn\?QH*l"
    ".32&W@mQYV:BgMBWwa32a]&/B&UfWF2:Z}ZdeB/0jNqwig:hL*@1GOibi4w^9o#/26XJ%W)::e]6L!u)]CNIURnI@)<q)Qf=fT0+{6IGm[zBY#[w^oR1O&tQ"
    "(EE103kpbk4G$7%F.GwAek2H2{q$>q9rb<RfUNeOcFG*H4fBSA}T<)\?]SE6z7eCS5f%P<C/DF+@c:wWz^/GyZ1mL+7g#6wu>MR}uaTWx$aLOOr0tDwc125Pw"
    "*%(W@joUi\?{.9h]{y^X$dFTbS1onxdl&^ks2nc4$2lVjb@o$0}^!:sQ&RQTY1)#v52l%O*@@A8f10lE(1p2a)d/eYu1]^#%5cRhe)![-)1psF!2[Uq<@%XL#"
    "/9zcv4Lna*}RRXE)F(wD\?t3*b1][r41#bxM]ayv082:^2{wkQ8SRlq]}$xB=&0fY85CyXNM&lZX]-T^S{V8t-0&MIMj]UM:>L+L-}2yT2@TeY7@}V=$Sr-E%"
    "0%wG41oe=1@@&A00S-8%0T5Dc{xgf>}uc)6}uc{43>-mq1QUCj}V3M$1o5e@@qV]>1Pk}m{wjBQ@RlZf0@Jfc0q#%500JA5@@A61}VcT2]85B]@qV)4}uErk"
    "@%5M90qD&{@R(@}3&{za@rAk#0r--l0SeFXFDb=K@@.Y+@}u(%@TNOmjqv8Q5:ef=<PkO=5CDXj@U(1L0s5Y64+pNM4HJQd[DmUB)D$2!1Qs#468z>5]Aa1O"
    ">Z-{E9p<^w]yaAY02CIB{2+(SeEH<s2M+kl[Bb&81relC3lGuE@R=0:76C9u@Tonp*xE]<!zy9z3e(/X}%9Ri)9>$w{6gN3[bH+%]Z:9EaS*F83a]S]5^E!P"
    "1PF>44G(zq}tZ=(@O%aNjP+@n}WiF%3+ulp}#c(<2L!i5}tw}n@@Ju7F=2&-}ucK00R(Y/0TF/a[CR}Y0R=-$}t*^3}27lW0%wx5{5s*w0s5Ge2K.P<4H@vk"
    "1ono5&{)3F]9CP#@@Ai92M1RU*X}3A2(K/f}#(]]0@-l2}Uf*/@o%N#@0&/r1Qs]1}U{:5@qcD)@qDP$0qV=#3i{>q}1<:!0r9302QUUbjN@EH[cO)/@0I=-"
    "]9bxB.r7k6@q.AB<OYw{UkW}!1{>5r]Zi850qDx<{wy=\?2LF95^/{6T)F[0-0r&l8}32:N]8[v[)!D>))J7]%2Ma=5/:4MT2K-Ae0R^8%1[z]$]z]3Z4=dr$"
    "(hza@0OW.N&qrJU]A1)I)\?k6%&$Ry\?1n0cq}TjpZ4\?5::0S@!>3(/9tY.g!g{Sh>he2bbJjOl6c1S[9=@{Q0*@pY{()eMIP@TnMe(*$F=}u^8>0SRx.0QZfL"
    "2)Hkd6zHhN1[}fS(hy+/2kFt!2)6=b}1TmB@0ic14fuUf77yfl]zXzf(GFDYQv!i.[Dw{/}U6%=5CTGj}t{Qc[cBQg)/@7W@ql&-Yy=ps3>fJH&t99$5bbX$"
    "3IT=6(KrTt(\?7n+2(yE&(i^B/0TeMa@O%e8jL[&/}u=x+/x.-0)KwQQ2JllU[=o%8K\?Ig*)b5G>/y)k])DKr700-e$/77]s@3!Jx1{2\?:q2PU61}p+C[6TwU"
    "^JvbY6:U8X1jcA@)D%fa3pi+(*P]PF!=Dr/}s2ifJ[S*[dPMSw<R%ft5pV9}o6aLu)y%821{bOF7v9vR1s@{u((MDh:(%2McJT\?!5-%(k9]l$=5cTG![etHO"
    "jpq5N@SSc87z3AP>l84=:(t8[9(3&+{3.RY79UB3*yBw]2nN1{cGTH<i$G^z{ZxW(8sNt10U$nw]vscK@Q5I+@fNc(2Q!Gt{v4}O6:sH7{$bk]bP=X32jI%C"
    "(KT%MHvNqf%k$l<6ANNO!+5wZ*zG/1]vz$vYb@{v}*)#w1@elN[-e9:kPgA!<m<6j(=wLy4K.bq{vYo]!Rg4sjQq2/2)fR%}ul]:3>AWC0rJfa1[hM\?@TnCV"
    "2ltbb5C(Fb@q(uV0T5l0%m(G@{5Li>0pYTs}s<{&0}W641o!/b0@Ab#0S@bb@@z3J}V=)101eA72MKL[2l:=c@%et$@ri6b]8vzh@@&oj0@h]%}#$^R0@}-*"
    "1OIh.1]mfX1[&9e0S86K0]i>U2N2XX}2pG)00hP=2MK@6(jzaejOFQT%n0od4Hyb<Zw[fm0q(:70$2d#2MJD>@@Jxd@qN-k2Ocw8/7QME[=X:*^\?.&}1n@8W"
    "0pG%}]A:W%}#l=${4mXJ^FAKi@RkmQ3IS#$2LNVQ@qMD%1OqD^2)Y]b0@@(n0}k%#3I-<50S\?P}@UUn8@15D91PFWS7Yij<3L0p33g[2:6CZMB1]V7m]A$Ta"
    "@RbjT00ibO6g<8ajn5y1*wf>r&T(1u]9UDc^cqc8<p7XzYYpjc)bdsCYA)u{6164&(9ZfY}V(Wq}{T%C@sF-z+s.M$0rp}v>N45B<oTcn+levA7fg[F2J$q8"
    ")Ik=<10eSLa{ee%8Q{q3F+epI\?jKm9)hnv<PZrK:!aAON1@54c)/%1t]V4A\?/9IiK[XB\?{]xXH.(J4gZc[Fse&kBvH*OTlWg]MHvjqFZ$/blpG)=&Zd1f*[/"
    "]E*mZ6CyJ47zDk.5=]#r\?t#y99{J:Q1NMZ1We*Ts+[hic)K!La@Qo:M\?W/F\?d8toO1)>Q9\?uBr2>/<IH[bsjh34.HE0Rk:k>i@P=]U.sJ%j!e11mb4t@qvfs"
    "T)CNc@3riP6aj9Q@){@q2KHTG{waz669Dm{&@fXS3kry4]csg\?}uE]-{o%z{@0$3FjNa0T4b&(s009/m[bJ^(0q3A^4GVA32l%[<1qAiO0SJDc3JP#7}tyuU"
    "0%Fl2FEAa@2[v4r2MK&p<l#QJ0s5Sc2<\?*a{x7fQ{x7c[5=isf5CLXM1rEaq{vmCQ@$kOr\?#Kk%@P<*&6ak0B]zEzJ)Ey6e3iQtf3KvaE%lGsD>(ZU.0rJ/n"
    "cNxfq@@JG9{XsZp0qCZL0q^e%0msh<jO[ES2l>3U0pYE:]z[*J8#tNc@RI#/>Ke(Z}R\?CD}$opU0lLSu0PS$R}Uz5<1n86x!=cA*{$2=!0se-m)*!:K)E]im"
    "2.z{G84(MK<obJ3]91^*{5j)Y)Hx133gXkBbn4DCanVoZ[DP<k3*m/\?}UpcV2(Bt)apmPM)hc0o{5K%]4/cb43K)4l6zqT#5c%DN]}z9k}#DcY0Gaj.jpH]!"
    "0$tbk3KM%K8UjNE@{fB=3EYp\?@0E#-1on8\?]9jEU0rz([0SS@w][&LH]yh[S}v-LJ{yM]-]BR-t1QaP12)f(52)x+L0}(%c2lUkp@TE(/2jz:57wFvt3JQn6"
    "}4JMb@%1qq{w$E11nqV*{-eLh2&K.E}#)l08VG%r}%jK\?2mRdq5y7m^5^DR>)/0sC0ShVW4eyhm/=J51jpIe+59>bs}2}(yjpRt#%nhQk\?Zwor1{Q{)3jdZK"
    "0rrG3\?u<w!^GCcx3h0={@1n(i0#QSX@%N]&3hrY::]@dV0RVA^76<Dk{vm2k[:VRW00.y02M2qz3*dx/b%AdB2l+.t>jv*n<N@l<2[var}-+h}{6/<L2LX2K"
    "{V@7J&%2k.7yGlH0V}m32lpa/.r}da%n7*I3*[bQ1w0!)jONkain-p15CL&g:o0vU0]A>[^fpr!}U2[n6-/AO8v3!{:Ppl9(>I[.cNm{&{Y7W9]8eZZ3<%ul"
    "6=cYN2I}fm]v-fF2LM&46XTG32(j#F@OwK)2L[kyeF]3n]VE4h/ynF-2NPF]@]ss:4<RX:4Gz+y[AiZm1)Dfc)+&<c88\?Z2ak^Ioc&t>wbLiFA[=/cx)/ZGm"
    "@]9LNa=HL@jp*q[}%:#${%x6w(i[IXiU7#]}1@g.}3^Ma&ud[m}#)ii2M>hr[ECu(}tyS}@M=LgbnpI5<N&-Z2(BVk[f4i02OE)z2LF>*0vE&*/9q@=U>XUq"
    "aPr5H{Y\?6u5Clm]82JE#0nN!N\?u%nK0T6tO@}c0^[C-q86:9Ra[bqXE1m(15<N03u!+XUk<[PA74J2oA7npg23I8:Q5>IQTjNp=x(JocI1p#BJ1#(eZ6U&Jh"
    "<nPF@}V#O<6-GrN9vIya[/L)R}sKvz2R:W2^bjmd3OHxp<@<fm^sH#I[E+%vgXHnk2LFbU(Huth5:x2*{0:R<3H)Bf>[0KW69/VPW+yPV{VB-}3ki9T)b1(A"
    "2+E>N1P3<b&{>[S*wgp8(/vQT(GpnO7:n\?!*%)OE/8tx>@s[kV&q:}cKnWV:jobY)/^uj:024f-0t<=h2GVrV4.h(H)IDeS4&2#H^c-sK}r]43(hZtwb[y}l"
    "gZBc:0R+KF)cTMB/9-Cp]AUBn]NVMm1{-lT&@ga{7P^Yn4-6Bn1Q1h75^EN>[F7j63h%SE]-ta4>*P$92])1!0Sq#&[b8.J]zfdv\?qv5>.@RZ11>spA+V1!*"
    "aR@/5{5$<t&RJ#Tb]vdm@T-w]joD8v3Jyq02[DU6{Zvx3!9d5y{xx^!}%T^^{yc^8{5:)&@@rob0q39(@@p4c3j#T@*6F8&{wbg20TFJd]A$9Z{w:H.5Clk1"
    "1}gbj{52a9{7lo^<Q7}}@r9r7@}3JW4fL.<2k508)!)7.]4A*D1owk&@Tn(a]bu<0{xIxd>(Q!(}$P%^3kAKN1n&C<3&PX(QK&dm]8OdR0u*P1jPt>V3Lr9]"
    "d6rEp]z!#@)*FU1{5>lST))4%\?0C1:)ek@!4BUyi@q3i)chtd:7V=2u3KNQJ<PCn\?0#q1Q(\?*4A[DxKl{t2Yl7Dz9X@}#Tk+kIhw{Z4Gc0T[#N74-cL.$nJT"
    "8Sy/r@tJ9u}$PTS1NLEn3ig2q[/cTg)\?3cc{vns=3juzc2OdZV9pTg29u2T<>k$z^}s>:d9Y8AAjO5ge4*RW#5Zf$I@SR=<0Vgeb0TuKy&q:h(@{YmW[CAp>"
    "@@&=A}s9XL6z}hr2mgg+]Bqrw]9CK32[3.b%lPrc@$b$s[+0Z(4jz5N)Fl(f>kt8&7Yj*J{w+ur1]XSc5=yp{1O-08[=dFA2j&l6}2*^9<PUz^@p2pr}vZgb"
    "2)o\?*}2f<(@S&-c{wU0&3(44[Z-iT*}VuP*1N24=jtlKq/78is-tp382\?PE!]aZ9:2[MY34X9ylbo58!a7:vg(\?[V(+L}Xx1n\?9Y67b!J03qab00i2P}S=qb"
    ">F+vDQA5{[d*+cl>>(Ic}jN6A)B\?+q9r(k\?TLK3{T01Ry3Bo*z}TM4]}@ELF6y!J]9[)w86tJMtbkRK2PYcB/]v(h1XyKgM[\?*)f.9dx3(dD<F@$CKWFDD+E"
    "[W[Axjq5Iq+J]Yi=LWM68W$i#}pr<@2M:=\?1[&nZ1{287&@w8\?FFO%*5cSEA2I4@D=j7P-3xSp32)]XxS3Irz8o*5yb{r)A}T+a9[CrIz(Bdt*1f9ZS3/*%a"
    "=\?t#+:t%4PahmFD2L[1]5x9MP&t0.d(DY5CfXE$%0Raw.[ascm.pKXPZU]FK5gE+<}vz9^3Ij.x5+(\?3McQi%}tK*njM$.!%l*i[8T4w60sfbn{6Z:6@S9O^"
    "1lxQ)0@IQv)!=n+1[}@O4heNn}2pE50$kef*yk8a(fO(Q0%OY4].3)*[bS[f@@rAb2nv7h2(BCI(g:SJ@qlQ2@%e[P@}^xe}tfjy@0JV9>[1d58V*0t2M<>$"
    "}vJJj2l%h*[bK9/1{Ti6]8nW>)e^ka{X$A}0pGp!@qlH20@&014l/HTjMM7/.Yo1c!d&w@02My+Pv9/y82%=&YxGhb5+Cw{74.\?r3A%4F</n*-)e2lf3\?JDY"
    "(<<1A(jkMQ\?UupG6EaLm3L<PT(>FQe2QfD.}10hB}4S=khS(d^/:ePZ[+&1IFZ$+O{2tOt1tRa$<NJ6w-S$S402umi/-it.(h7hC1q\?WWbkZf(8V${6*9NHk"
    "&ybc-[VI/\?]dp*K]GyONjoT]o[6e/T7u(Q![a(8X6:@zm}0XHt67C.+80x09}reKz)*Pll=&:yC7V#tx3iZ!80vdW@)^Zi8)is4-+{A&q[eTuT+g!xc2Gl$K"
    "MHKu]}s1+w51&+9].Fj)<0z(1X7pRS).eDH0%y7G]-&hR3gmVWiW$^m{a1&mZ{#l7-Kuj\?/vyAF.q>4n59i{84fmmm]+WY0/c7.44mX%QjMU\?@@>NZ>4FZR7"
    "(iNm*}#cH0}TTar3F>Ks4epbw\?QzI[3k9vx&}A3t1Q>281@N{N[^r}+}uu%60sw(j)EnV>0}vf44ew][{6y&z4FuMG1[\?M@}[87*(H1SB(&U#G]XfGA{XT*>"
    "3MW<*1qI:\?7YsZtd*ljO4hxS=&]/dg{4Gi}68.>r&%(1[@qu@&1R7272[lT$@S@(e[NQ}ijPkAW@SAb#1]!P5\?2^MP@q({4K]UW+@r4SC3h+bb}3dic}1BZ^"
    "{680m4cvJ[0R(Y%%nriD@q(%9{Zl-$4f3e30%!=82($820%n8$0r992@o%Z&3ighd2Ngqs3d]QZ00&r2@qD9*}2Hr/0rJxf1m#Q61o{dP4fDG<@qVS#2MjS9"
    "0T!S4@T5&d@S0AeFDC#N0rrJh%muf<2#P=-jP2D.}vA9600Ac4{Yg^2)=jlK}#Mr.{38\?-@pZ)m{XKZB)azmk6ZCqW3iGnO3d/p-0@@9d^eS.p2OMLa(<:0)"
    "(*J{s0R#oe@15(r0{*v6{x\?5$0pf)/0T/tB3hL803I%+%2*&UV*WkNO-khGNF+GuJ)/Jy)@Q/:(@@\?6S@T5So5a6+}67k#q2(TY84KH/q^!iwD0VgFd20BuO"
    "jokiY3h@GT]8O:S]z[:*2M1r3@jh\?r00JFW2mqjm@r0w{7Y<6(7xBTR4aj+N&06R{1L01j0pYKu1kR/i*YEPZ}0X7.}tHMG{X*Y=3\?%Y21pB}Z2l9cT3i5#["
    "0T]ek}#M%4{5qe#%m(A[}0ELW%mb:x{Y]^P0qDzx6A3}/}UZG>3Kckq5yy%*1odl!>i@AS0%Fr01owD61N5MwjPs%C)eGYP7.4#4&st8O9o3I&(=/aW}U/{K"
    "5Bp>g9(4WF{5LagQXUA1{+q)W4&u2v5D]Au}W8M\?4g9vE[Bvz42)P(k>L>vd[b>2CZ%Y1R2*1gN6DdQ]<Q@3>(J!11@Fy^M]869]@oaFh{Xc0r@RDN<]z[1T"
    "b[hNh.sZLe01Y54M&O=dYHR&j].[T<1(Py\?}ty9\?%m3%F{Sz=UjL]-n0qWub{wsNE2)Z824iCDT@tCbq3>07l@1n[>\?#JAp2mQ\?e0%4/R>(.4}FERbS6WEy6"
    "2)oR*(<ac#]78!^0q=oX3<WcD8TcD8)=>CT@$J]K{6x%Y1N$jy3KbL&}4sky@08-*]XPYI]cS2\?76aa5}tyG@9]M$[4eG>W3Jhdw83mL%d.bAR8W2{t2K.(u"
    "4!FPL[=/vt0FrOyjoCr-835a*>ly.u&}-xR@q3y2X90#r00R#>2(<\?[1oc{y4GML56-wK-&TkCV%mt<^4.@5K}$U*p[Z[xq*xvD=0(B%q0S@kV<]%ru5GIwR"
    "{z<$73jVb368Z.&2mg>k3KV>l{5a%^)^P&S>[Tyq0pfZo]xu8\?0%PhC1o5b{0S-Yh68N*[{#.2A0S%kE<N&Sy0rzW1[+8.w}H(wAjOPkl<N8:y>(HwT1q7[0"
    "/-HmM@tkFLL>kd.1paAs5=\?s*2M$tt0q)w<{XCl>5!=}}]PGQY@Ps$u=H}2h]A>Wh2i>%W)f@^{}uc##]5oVp1M]dl*6NrJ3iHwn[:(#1+Lqh5]-Lvf}3ZcT"
    "5cyk+4E:D068\?}l1>A[<1ip/f(Gw(({w>3$7y.{47X!p60o@H3{X>0+:lTif}24^{jOGlE{xIMc0$U!C)f:l@5FUrR{X>6rQY7q^F-=\?J&}0cS5A-JX&Uqm&"
    "4f(Xe@@At:4*iNU2]JWk}#ur]{XaNy}XoIK}%s{-bJ5+7{X>l#1{2eA\?t^&Y2)H%41][U%Zw=Z^0rrAj{$USv6Z#503h1@83jl[f)F#kL02uWD00&rz]8Gyc"
    "{y4@s2L5o54/chh0SzY-6:9p>5r/\?.jok/t]ca7>0>[Bf0Si&b<nN{D}uui=@{*l%Z7QX>0rzvc5D&R%3iQzc}tRcx}t{9O]-U)f{y4fm1}7ke0PBMt@0JA0"
    ">dgY)3*vrl)FVz%}uW=o0@-e.69[:30SK874G$!@1n@Ae{#h23{XK<R2>N:8f97[U[en[C<PMy^{5:vF\?1.9N>(\?T(}3Nx54IXQl=jZi:0Q*c*1Kci$jL5%l"
    "1l8A8=kVDjb]k>O69XsD@15C-6Z=W*3I:hn*1lU[+>QDH2NoR@4\?e&([^-IS}w/LL2d0H[-3<sf&T(Rh[:$M>}sQc(3F3r>(biz/8S8dQ4g&yj6-^!l7.C^d"
    "}2QY:/c78u=d}1k774}sfxHp-/BeC<75F:>*YQp8[GMS31[JG095Oq-=4j{R@MU=[1s8BP}#l6>8O+qpjP(G1@$25068\?+i@qmz\?1ps@h008=&XC=QdF+o3H"
    "}#cK7@Qf*T}W&-b0@.S(2km//3&Qef{Y*^5@Si65}ucSK@%x2t[+igY0r&o1@RM:9{X$i%&So5P@@Ab/1PeC#}V2*V2(aM8}t/ZV4/DFf0qlr)0SJ0b]8XT{"
    "@U=4s{wa:<1onA32mghp0008)2M>Oo0rSe$3iP!$1Wkfbjp802<{y$S&rG8Q}SXAs>LB@-0rg*I\?t)VL@QQ*n}uM*1@TwY7%k<vV1otIn3JnrB@R3Zd<OPb\?"
    "2kO2$[=OT=00&-n5B{y&5dOHp{YpNa)edg*(JEU3(&vs=0SJR]3JZa*2la/5)dykO@}V-${4Xpw@qu^1}Xn(j@S}G={vXsS0qDi*3&xL}@R)8$2]Jvo&0eoO"
    "1Qkne6aV).jEVlN<mtbd{yFLu2(jD7}V)9a3gdSV2<qMG4hf&S4c[i6}WSuj*ZT\?{5!:Ey*wSl02jQ:X0SRVM^/{lw3iyzq[7EEx@%wrk{3@:*@)7c90QZ8w"
    "0taU</Y}]610^<w>M7[<].F{A1]Ob^0o$1j@<I}A@}#3X}1B\?c[add[}2gcS@p7^o0WWyb>GT7#)cjMX0SRSS)Fdma1zmi>jta+O:PAgG1R7CesXOz!5C(>f"
    "&]{&Q4hL=A3JY$&02sI$hWpOU9rOdh4\?Y3q(kqFFgwZ9-7s}yn6.To)\?q)GT4!PDcb]3:0[3:yV5DqBp[C%3U2<}%S(j>z(1RRgl{t#}vbn}p9{w%Z#5e<ah"
    "coQ9z}Oz*F6z&s7@RbxG8Up/=<ne-Z8uIr91>\?Pd^!<fY3h8^L*3F$Y9^2<Ej]CuV@qc:4@@-8{1]mY\?@q#%200-z+0Ta8C0r&=7@@A&k{5Uf*{X@}y@rA92"
    "@00Aj}tgo]0%Xk[2lkbg9qPvn%m(G&2M>280@h=%0@&z$@SAVn0@}#]3gE([1hU3m}tHEa{5z#t{xH*5{v[Z{@rAx40S&2{].[{X1Pw2:4GVK{0%!l02lsG9"
    "@}=A&]znQ/0T5Dc0SSxq}b1\?cjLYSB5aHej>lIwc}V)An!yh#e>MgUu[xzvc2p72Q*y+s}@TZBK1*5vE4hF:P{6.Gc&207i3KbU80Vy[E1J*cc:n+[Q]7I2("
    "@)glm<(#pr43]SX0K}QF[AoSM2&(Kc2hY}0%nhAX{ULlc@TXawa)G:b}X5k(0lb()&pP4#9@+1J0{q0S%na7]%fIEKap[t@2J#rA5/8IO^@NEPjqd$)2Jb^W"
    "%dPh.%m(=U^guu!{5C6\?3*v@.[B(wR2[lRx}3dh!@{*0^68hOv1pkFm2Ngzu)G@)-{.APs}Qs#!0soRq2>d(Z2&#fL5c-z>}UQW5{45Q=3>JNH{Yhie@}#()"
    "49[!J]B}fn}2]<=0@992{WOa(]Z8[*0NgDh)EQz{@qMr}5d!Zg4/uCd3iHLq*yS#!0SID}6MN2ljrB$2)i<W]3\?J((]/0GM[97O/2ndEh1nqyC{UMz$@qLTn"
    "9o8C:c<FYg(HUh)2No!!{xZu\?3LR[q7ArXQ]5Hz.<Qr3%1{>Rdb]d6A{y=@%5A%P7}XHsq{tC/t9#(41}1sX98$:4)7.mFG}tpi@}V3#Odc*E=&0o\?}5g57!"
    "@UM3%7z3Gr9%rL!2I6Bw)G&:*]+c)q@t>./{3^cVjpR>B7vp!Q8qj&&{X+{8\?3aS9[EUrY0rrn%O+r/{[b8>{0TeU=5BGq<3J/rd67tw:2mgOB6b}ev]-Bs$"
    "(&c^M0rA]v@pHs381vy)2L!Jp3H[l4FE@v=2)gp!5af\?Z/AR9p}TjH[<oBlC0%di:@pxyYcLVK:}W%tC}vSz$0q30#}#WPl2nEKk}Vdch3IzNy3hT8]5=i1p"
    "3P*$Ljp8[T}TUcSb+L8j}TK^g)Hxg1{5>3<9S#-IFFvU\?]a7Wj1nAz}1{j&00R^330[^ky(IRL4^e:HP01eMl{3.tv@1YXz2k!>{10vvy)FW4tN/&]&0se-7"
    "2lUnM@{G)e2kF2<1)k{@]bm093j3.s3Jo/c}1>j2{waj!}{U/z{4[K<}UHA[3J*tn68h.12m7Uc>N3+.1}x#125gV{jOG6Y1Q1&a@@A9h2(Lqe00Ai9}40ui"
    "@QG%[}sBm+67(Rb3gv%e@@@c0[+q>H1ptz1]7@aZ}3WAc}2y)8@r:bg@R(Z01OIrT]9kg01[\?u\?2MTV70$$$I]9>l=(hQbQ%n8-[8#6aQ6BU9Q}Ux:O\?#A&R"
    "@q=W10%]kc3iP#42oSWA0!gmr}1%s!^Gom(0TF(r@qWcb2]9NN[D!nsjK:Sn.1z7t}uEC)(Jc[Ued.aA7v}ub7qP\?t}TuSw9#UyZ]=0b)dDgx}{8i$-KPU:{"
    "4&L3t&}IN7790wK[D4zO5C(8q6Z(DW@quY={w9tp0Nn{)@NRXHWF6j^-}Fv0[+SjG1}zjq2J(k)<PCtT2j8Z@0sGn54eZ8s&R=N/\?@B\?B/7GW*2$+<t^B/oH"
    "]y-!T1]v9L=(&&253p+GjOv<@4HRa&&U%9z@o$v74cEu@2Nf+2}1B^Q1mUZ#{uRzE<N8^&2)*IJ3IaSn%0wGZ3I9Y<&)[}:8$age)H[H+(c{r@(iuF^=\?)\?F"
    "1j)UZ}Vc%21m2W-}q\?+V3\?}Qv2){>o1{tCl{6Ifw3f}:e1nqAW!dq&<>j0{d}0^}F8y3i#)*6A#}Xfnm%lf:]}u2Bx@}D#%49Dv76hqc:j/!mokfYcpki8NG"
    "Y&D.^k6rOjkBa=Qk1MbAk0w!]j=iC3Y+N@@Zd[D2k1^Kmj.uj6Z69p6jX7{oj#S*0Y/Oa9Zb}MXkr5bsj(t=8jWVV*j-%AZk4@j+Zcw92j$-QJkj}lukiMQu"
    "jw6.Wkd%i=Y>R5Zk9}*Xj<o*.j#v%>Zb{iEk5uc:kdva8ZF5Q1ku]h0j#*y<Y}>VXkD#]dkj\?-/k2SoTj-BH4Y*d1IY>fcmY$KM]Y[/ZWkc{VQj%/][khk#^"
    "Zk%WJZoidxZ8Y\?&Z6}ahj]01gjUWCGk8rV!kfEqMkjRObj>mMhY!4e<ZisQ/Z6rpZj:.%7Y#t[LZma$uj%je%k2eQEZgVQjkDgb>j)p{}Zh)u!j{+!<k42La"
    "Y-Xwmk75Y.Z5/y[kin#!j{ZjxZi1GAki:yMjPGNdkmgu+Z9wQGj@K4%kJ6k.kcF7xYA^N0ZORyaj.&i$kh5=SZfjD\?ZiU}NZoL]^ZG#v:Zg0M-Zq)K(kaQQ:"
    "ktD[Nk4YXfkcSf8kcLk.Y@iD-jUD2lkd-ghkj63oZr+-#j+9:.j>D+[k1IdAkhHU(kGI3dZ5Ki\?Y%[ZWZXL!2jXg/lkhe^5j@ye=Y/CEpj/gN)kAwnQY^9eW"
    "ki/H9kb[)zj+()<Z6B@JZo*:tkkPd&ko}Qej$J(VZz:.PZvDP8k6lv]Zi!6aZ9j9OZsaOskrn3Xku]axkeZXhZxH3Wj>x3VZz7+cY+2i7koq<Bk0G%kjLd%W"
    "kfDwjj@@R^ZYezwZt1}{kg.syjSujvk1>c8ko4fTkwT#&jo^htZtZ1MZJH}&k2OBRjp@}>8VxdE1PmWq*ZjLv5hi.i@rh/<7t/q)>i8%O}XFX(]85SH5B]=t"
    "+L\?+=@S&!o[alFJ0pQ#\?2@}L[]z!N\?ndjRT]ZTK69QAOhmGqL[1(]}a[EUTe3Is@4[+rn]3lfl*&ud\?12>X3aF+Wz02NR7q1p++I0s[&o[Dedu@#.^M@}MG{"
    "(K<r%1pBw@l/u(91]/kl)Cw9j{g%$)i$QLt73<*m0Us&i0sORMi&kK@0%^-P0RuB9zW([9=\?Who!!7gN-cflt2[VMR6BK&\?8r\?k\?69c+l6==3W10DId]vTGU"
    "bmA@i@u9r=)=wxJ6XTe)%m=GVbM!bt+U+kybeslVe!YuB2R#=c\?2%hk4YHxI]va(!7-k^a3iB@Aa#EEz0q3T1=<]mc=Ou4f+NadBch4/A9PN}i2UAF@jN\?C3"
    "=ia0&0SJ$!9$[J3{T63$}W9e#3ln{y>[THo7Y2G\?@qk2.[+SdB9$)@$0}+bu6:%G=<]<V*[WwL>2l1b}K]UW&[!PT.[Gv-aFDU462OmE066f673\?<uH{4x2."
    "}RRI64a$EQ@tM:PBnYqD]CN2$3Jo8C[^c644eGJo%k9nI9%qJB66Zq41]oUFOeSNq76Bgc1s1rW2A/xrjoVJs\?4GRw1Nlo{2NZIs/+7^4{wL8z@@9Sg5-Sry"
    "1}ywq4Ey\?t2/EM5>j]i2)!uI\?0VHUe2NyRM{Zl#<{vm+q4Gl4@1{L24}$X5r(Ih!e2kwRd0R=3&2LOAb3E]RV4\?G3v*u}rb@{H4e2{6(<@Q/+12MJAX<QZ8I"
    "{z-5\?/6:4l3lmtYb[yyG0]%<}4d1h[{x}Ar:]Sgy0xUYKjpzbz1>T.a2NG\?>}%aaV6.%KA5C+Wz4I4ClXeU5G[9xrs)f:*e@MCJ8b(!550r@F$2>4/<2<.&:"
    "4E<#91oG2t6Z(Xy=Hyz{@}XXqbN.]{6yVX$@0%Fx[ECd2*Aldy67L8/3hIy95fgfq0n(+-1onSd4/OG=cGM0g@OX\?7apW#i@m/YJ^f6E!&uX3Z)/9y@1O@38"
    "7Xw1J]J#[\?jPU/#{6Z=${4XT&0%FA2[=/v3@r@:S}t*^25Cch00%eA41]/ql03\?KC1POJ3@qM&{@}#(%0r@Ma}#==]2Ngta]-s)/1{a=4}#MH5{wUu{]-j:P"
    "0r8&$00ih@0RMQ10rSM80Siug@RDG@{6Rrn@@&e]}#M9/}#uf!@sOM1[=eW$@%f.v0r@P40SAu9{Y7u(}#3T400JM50uqaDj]+xr%lx)={ZMYY}1jK\?68qFJ"
    "2kv/%@SZZQ2P-^H}$Zl&(hRg/}oeQv3ipF5@rAtR@}#G/0@091}RhzO]7-s!1nho&*xd3n@%6eib>[Uv0Rlpu].X{%{YQy10UaN)@R#M$@Sq9I0Qx)*]2g.a"
    "1nq#[0@%8p1m(@<}mWao5C2/3}%:yX@Sz<0@V6\?)(<SK^3h<IR8U[{[0P(PRj*1LbY{6v/j[J/Ik9zz&j$iv/ZY0)TZYjum";
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
inline bool bcNet() { return botNet().inputs() == N_FULL && botNet().outputs() == BC_OUT; }
inline bool duelNet() { return botNet().inputs() == N_IN && botNet().outputs() == N_OUT; }
inline bool nnAvailable() { return bcNet() || duelNet(); }
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
  void duel(const State& s, int run, int blk, float* xr, float* yb) const {
    float f[N_IN], q[N_OUT];
    duelFeatures(s, tr, run, blk, f);
    botNet().forward(f, q);
    solveGame(q, N_ACT, N_ACT, xr, yb);
  }
  void policy(const State& s, Rng* rng, Gene g[4]) const {
    if (bcNet()) {
      for (int k = 0; k < 4; ++k) g[k] = bcMove(s, k, rng);
      return;
    }
    int run[2], blk[2];
    roles(s, tr, 0, run[0], blk[0]);
    roles(s, tr, 1, run[1], blk[1]);
    for (int t = 0; t < 2; ++t) {
      float x[N_ACT], y[N_ACT];
      duel(s, run[t], blk[1 - t], x, y);
      g[run[t]] = actionGene(pickFrom(x, N_ACT, rng));
      g[blk[1 - t]] = actionGene(pickFrom(y, N_ACT, rng));
    }
    for (int t = 0; t < 2; ++t) {
      const Pod& p = s.p[run[t]];
      Gene& gr = g[run[t]];
      if (!boostAvailable(s, run[t]) || gr.kind != K_THRUST || gr.thrust < MAX_THRUST) continue;
      double toCp = std::atan2(tr.y[p.next] - p.y, tr.x[p.next] - p.x) / DEG;
      double d = dist(p.x, p.y, tr.x[p.next], tr.y[p.next]);
      if (std::fabs(angleDiff(p.angle + gr.rot, toCp)) < 4 && d > 5000) gr.kind = K_BOOST;
    }
  }
  void act(const State& s, Rng* rng, Gene out[2]) const {
    Gene g[4];
    policy(s, rng, g);
    out[0] = g[0];
    out[1] = g[1];
  }
  void rollout(const State& root, int team, const Plan& other, Plan& plan) const {
    State s = root;
    for (int d = 0; d < DEPTH; ++d) {
      Gene g[4];
      for (int j = 0; j < 2; ++j) {
        int k = 2 * team + j;
        g[k] = bcNet() ? bcMove(s, k, nullptr) : Gene{0, MAX_THRUST, K_THRUST};
        g[k ^ 2] = other.g[d][j];
      }
      if (!bcNet()) {
        Gene all[4];
        policy(s, nullptr, all);
        g[2 * team] = all[2 * team];
        g[2 * team + 1] = all[2 * team + 1];
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
  }
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
bool couldCollide(const State& s, int i) {
  for (int j = 0; j < 4; ++j) {
    if (j == i) continue;
    double reach = 800 + std::hypot(s.p[i].vx, s.p[i].vy) + std::hypot(s.p[j].vx, s.p[j].vy) + 2 * BOOST_THRUST;
    if (dist(s.p[i].x, s.p[i].y, s.p[j].x, s.p[j].y) < reach) return true;
  }
  return false;
}
struct Prediction {
  bool valid = false;
  int k = 0;
  Pod p{};
};
}  // namespace
int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
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
  if (mode == "hybrid") S.P.oppShare = bcNet() ? 0.0 : 0.1;
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
      p.lastRot = p.lastThrust = 0, p.lastTouch = p.lastMasked = false;
      if (tk.turn > 1) {  // same rule as the training data (src/dataset.cpp)
        const Pod& q = tk.prev.p[k];
        double c = std::cos(p.angle * DEG), sn = std::sin(p.angle * DEG);
        double ax = p.vx / FRICTION - q.vx, ay = p.vy / FRICTION - q.vy;
        double along = ax * c + ay * sn, side = -ax * sn + ay * c;
        p.lastRot = (float)angleDiff(q.angle, p.angle);
        p.lastTouch = std::fabs(side) > 2.5 + 0.01 * std::fabs(along) || along < -3 || (along > 203 && std::fabs(along - 650) > 3);
        p.lastThrust = p.lastTouch ? 0.f : (float)along;
      }
    }
    st.timeout[0] = tk.timeout[0];
    st.timeout[1] = tk.timeout[1];
    st.teamBoostUsed[0] = tk.teamBoost[0];
    st.teamBoostUsed[1] = tk.teamBoost[1];
    st.turn = tk.turn;
    st.winner = -1;
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
