#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")
//   "auto"      hybrid if a network is embedded, otherwise search
//   "search"    evolutionary search only
//   "hybrid"    search, seeded each turn with the network's plans for both teams
//   "nn"        the network alone (two duels per turn, no search)
//   "heuristic" hand-written baseline
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
constexpr int MAX_THRUST = 200;
constexpr int BOOST_THRUST = 650;
constexpr double POD_RSQ = 640000.0;
constexpr double CP_RADIUS = 600.0;
constexpr double CP_RSQ = 360000.0;
constexpr double FRICTION = 0.85;
constexpr double MAX_ROT = 18.0;
constexpr int TIMEOUT_TURNS = 100;
constexpr int SHIELD_TURNS = 4;
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
  double angle;
  double rad;
  int next;
  int passed;
  int shield;
  bool boostUsed;
  float lastRot, lastThrust;
  bool lastTouch;
  bool lastMasked;
};
struct State {
  Pod p[4];
  int timeout[2];
  bool teamBoostUsed[2];
  int turn;
  int winner;
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
struct TrigTable {
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
  if (tx == p.x && ty == p.y) return;
  if (std::fabs(angleDiff(p.angle, p.rad / DEG)) > 1e-6) p.rad = p.angle * DEG;
  double a = std::atan2(ty - p.y, tx - p.x);
  constexpr double maxRot = MAX_ROT * DEG, twoPi = 2 * PI_D;
  if (firstTurn) {
    p.rad = a;
  } else {
    double da = std::fmod(a - p.rad, twoPi);
    double d = std::fmod(2 * da, twoPi) - da;
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
  double m1 = a.shield == SHIELD_TURNS ? 0.1 : 1.0;
  double m2 = b.shield == SHIELD_TURNS ? 0.1 : 1.0;
  double force = (nx * rvx + ny * rvy) / (m1 + m2);
  if (force < 120.0) force += 120.0;
  else force += force;
  double ix = -nx * force, iy = -ny * force;
  a.vx += ix * m1;
  a.vy += iy * m1;
  b.vx -= ix * m2;
  b.vy -= iy * m2;
  if (dd <= 800.0) {
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
    if (p.passed < tr.total()) p.next = p.next + 1 == tr.n ? 0 : p.next + 1;
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
}
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
}
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
  float rot;
  int16_t thrust;
  uint8_t kind;
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
  double angleW = 4.0;
  double blockDistW = 1.2;
  double blockAngW = 3.0;
  double timeoutW = 2500.0;   // per turn a team's timeout falls below 25
  double shieldW = 300.0;
  double speedW = 0.0;
  double oppShare = 0.3;      // share of the turn spent evolving the opponent's plan (0 = trust the model)
  double rescueSpeed = 450;   // blocker's assumed average speed on the way to its own checkpoint
  double rescueTurn = 10;
  double rescueSlack = 10;
  double rescueW = 3.0;       // per unit of the blocker's distance to its own checkpoint, at full rescue
  double rescueAngW = 6.0;
  double rescueSpeedW = 3.0;  // per unit of the blocker's speed toward it
  double rescueLead = 10;
  double rescueOf[2] = {0, 0};  // set each turn from the root: how urgently each team's blocker must go (0..1)
  double boostBlkW = 0;
  double boostEarly = 0;
  double pathW = 0, pathR = 800;
  double blkPriorW = 300;
  double valueW = 0.5;
  double (*valueFn)(const State&, const Track&, int) = nullptr;
  double runShieldW = 500;
  int rootBlk[2] = {1, 3};    // set each turn from the root: each team's blocker
  double snapP = 0.125;       // chance a mutation snaps to the top bots' moves (-18/0/+18 deg, thrust 0/200)
  double boostW = 2500;
  double blockOff = 1000;
  double blockLead = 1e9;
  double blockStick = 3000;   // blocker: keeps defending the runner's next checkpoint unless the runner is this much closer to it (1500 in v4; 3000 = 57.5% vs it)
  double shieldUseW = 0;
  double robustW = 1;         // weight of a plan's downside against a second (heuristic) opponent prediction (0 = off, 1 = worst case); doubles the cost of an evaluation. 1: 76% vs v4; 0.5: 71%; 2: 60%
  double reactShare = 0.3;    // share of the turn spent re-evolving our plan against the network's answer to it (0 = off)
  double pinNet = 0;          // pinned runner (an enemy between it and its checkpoint): play the imitation network's moves.
  double pinRobustW = -1;
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
        else if (k == "valueW") valueW = v;
        else if (k == "blkPriorW") blkPriorW = v;
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
inline double evaluate(const State& s, const Track& tr, int team, int depth, const EvalParams& P) {
  if (s.winner >= 0) {
    double v = 1e9 - depth * 1e7;
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
  if (first && P.blockLead < 1e8) off = std::min(std::max(off, ul - P.blockLead), ul - 400);
  double blocking = 1 - rescue;
  score -= blocking * P.blockDistW * dist(B.x, B.y, tr.x[tgt] + off * ux / ul, tr.y[tgt] + off * uy / ul);
  double toRun = std::atan2(O.y - B.y, O.x - B.x) / DEG;
  score -= blocking * P.blockAngW * std::fabs(angleDiff(B.angle, toRun));
  score -= P.shieldW * (R.shield + 0.3 * B.shield);
  if (P.valueW != 0 && P.valueFn) score += P.valueW * 50000 * P.valueFn(s, tr, team);
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
  const Plan* alt = nullptr;
  double scoreRobust(const State& root, const Plan& cand, int team, const Plan& other) {
    double s1 = team == 0 ? score(root, cand, other, 0) : score(root, other, cand, 1);
    if (!alt) return s1;
    double s2 = team == 0 ? score(root, cand, *alt, 0) : score(root, *alt, cand, 1);
    return s1 + robW * std::min(0.0, s2 - s1);
  }
  double robW = 0;
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
  std::function<void(const State&, int, const Plan&, Plan&)> model;
  Plan prior{};               // the network's plan for our team this turn (blkPriorW)
  bool havePrior = false;
  double score(const State& root, const Plan& t0, const Plan& t1, int team) {
    ++evals;
    State s = root;
    const Plan& own = team == 0 ? t0 : t1;
    double cost = 0;
    if (havePrior && team == 0) {  // our blocker's first move vs the network's
      int j = P.rootBlk[0] & 1;
      const Gene &g = own.g[0][j], &q = prior.g[0][j];
      auto cls = [](const Gene& x) { return x.kind != K_THRUST ? 2 + x.kind : x.thrust >= MAX_THRUST / 2; };
      cost += P.blkPriorW * (std::fabs(g.rot - q.rot) / MAX_ROT + (cls(g) != cls(q)));
    }
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
      model(root, 1, myPrev, oppSeeds[2]);
      nSeeds = 3;
    }
    if (oppShare > 0 || !model)
      theirs = evolve(root, 1, myPrev, oppSeeds, nSeeds, clk, clk.ms(), budgetMs * oppShare);
    else
      theirs = oppSeeds[2];  // trust the model's prediction as is
    if (model) model(root, 0, theirs, mySeeds[2]);
    havePrior = model && P.blkPriorW > 0;
    if (havePrior) prior = mySeeds[2];
    int myN = nSeeds;
    int run;
    bool pinned = runnerPinned(root, run);
    robW = pinned && P.pinRobustW >= 0 ? P.pinRobustW : P.robustW;
    alt = robW > 0 && model ? &h1 : nullptr;
    bool react = model && P.reactShare > 0;
    mine = evolve(root, 0, theirs, mySeeds, myN, clk, clk.ms(), react ? budgetMs * (1 - P.reactShare) : budgetMs);
    if (react) {
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
}
namespace mpr {
constexpr int N_FULL = 48;
constexpr int F_LAST = 44;
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
  f[k++] = known && !P.lastTouch ? std::max(0.f, std::min((float)BOOST_THRUST, P.lastThrust)) / MAX_THRUST : 0.f;
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
}
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace mpr {
struct Dense {
  int in = 0, out = 0;
  std::vector<float> W, b;
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
      double lim = std::sqrt(6.0 / d.in);
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
}
inline const char* NN_WEIGHTS =
    "o&a.x126.F0h6s+0h6q%0a90ujsIVF<l3Bk2ljPC@]:E-1o!]c%nS5/2lj8%4fVO16hs52}T:HW0RuN3@S0Gh53<QKjo>Pw0jhX90009g%nJ92}#M&$0095@"
    "0%wP6@@A31%iVt@km4.#00Jl20rila1>fKWj]TZz3<3zd@@z/(0r&l41oFM50qMP%0qVS}0SJo7@^<za@SAl30q#{00rii81BnV}jP4e>FJs5n0SI]K3Kl!S"
    "}t{Z5}VMB27ww{G4e{[0\?T\?6T}V3P%0se2>@}$%t(%uJBjpJRegv<&k]-BJi[dWIJ%n8-]0QP{Y@Sz({@qMx$1%s6I0rJo40%!M1{Ypo[@yIb)j+i>pebfRx"
    "009if0q#{0%nJ6100006@@A920rA910S8@#0S-o300if30SJi30BT9Djp6Kfe-=)p1{1L>0qWc00rro30rSG0F-vpO0%NY%)/AB>%nJ9200rb$@@9950&Rcx"
    "joC=s8Ur}70rSeS@%w-98W3iB0%5i9Ly.ln02$O=}WSq#0T5A3@@R({@qV<00(5hTjPCG/@Ubtj000aQ000310000000006@^+at0r8#[@@A910000000000"
    "000000d{:Nj+iRee.71u0r96q%n8-%@S8]#0rAc400Ju500i2%0q#@)}t{V$00ic20rri40>DHqjoA}q^VTUj0SSh[1o5f3}2QG{@@r5(b7x[!{-Rvy}uvf2"
    "0S-r4@R#]#@q(&$0r\?ZfjwCFJ0@Ai93h#bgFbO9I3Jye50Vg*$eD2FG@@}@<01x87@q)8>@[%J)[br1:{6fmwjot6]2R#FZ000h%0rrc30rAf30ri66J!c=L"
    "1[Jh{@rrc2@q={00rr2#%nS60%eZ{]jocJdiUIq)0rJ2N0r.PZ@@Ji3@@Jc3FDsl}(&l.M@%5A10ri30@@Al60rr930t=GyjPB%B(f56E@R=@{0s5k}0r&J6"
    "@@z#>^F.eD)E+%naQXtN0SJf20q=M)@@r5#03jY/jONj-U*@4m2ltN8@q$BG0@&D1@l1QeCWt>j8Xaqr{zaOD@@q{01ofOV00Jo61MES^js7Cg4fbDU0SSru"
    "01Gkka[=lA@S0x8@[^O-0}D3+9dYG>)F)1X@r@/f}$/Tr(Z}0zjPCGY]6(XW008]@0ri30000000002#@@r610000009BZY4/(.i000000003100O:AjoN8C"
    "@c[.Q0%exC1N+jH0@@8#1oF-b@SzY)1oFw%6Ao9E0qu=%0T58%}Vvf23<d4FjP+&U@UCLn008+A0rri2@@r6000i66@Zl(a0r-4.0rri50rr91@@r30@@r63"
    "0([T+joy8Fp7)H72M>dV1oO2%1Qj#8{ZdeYSijC1Q4KdSOb@Q100A601o5Pa1QTO.{D>05jPUJY2[l[n008(%009300000000004SE:*o0r8@%@@A9100000"
    "0rJi2000300qy!ijL&-Tm$y<z@RV-W1pCCk2)ZUu@%5Gq5ATf63I:-54GBS.4H&NB}UZT31oez#%jdY6jO]u&7m$Q70q+WY0@&b%1n&o40%[&f3Jgbf@q)l5"
    "@{x1A4GuLb}W0Ji1+Chsv}>ZVjrjNF)iq2\?1n-Yd{Yg^1{5Ur@@R=H20rJG70rSc1@!=pb>M7\?=0Ubzo}#M-]@R^+sjo$GO1][-60SS!v0rr320%FG40Srr7"
    "=41Xt<qwU<}u3P)}2HM}@@J9100i3301[yLjHla9&2hT{00i6H}%<jG2(:/90S@xg5C=q71][J43v/cD-0u.J0@@r3@Si2%008CXjN>uu0Ff.P00Ai30q#9-"
    "}VN330@&ri1{1&d00ru9@{Zo$3J72a0SA2$0r&J50O1-{jO5WY@$LLn00z)H0@qG\?2MU2d0SSh[8XibV0S&o6}SSN)\?T{Hu@@&i20rA5$@>dKljojH9/W*65"
    "@@rh%0rJu60rrc200933E%-N&5dwZD)/rp!0rAc2008@$000630Y[#!i$AX5.akdM2)fco3iSVG4*Smr}#VTH\?%9k]2kncm9V9@.\?1ymt0t2Oq}U#4Nk%ZcD"
    "jPLM-7xTTq008]@00ic10rr9100994F-#U5}#uD>0093000000009920003108RpQjO[<P]8nN*@R#@@0SqfW1PnA6@S8<10r8&#000c43uH+L{5+f}02+!q"
    "%nR@>}&JtCj4aM:/aEDD2(T*3IRE}f-oWH2}uM(/6eZ:j[^+f(3!auH1pa(a@tbIm@R#iD($Z..jpHB62C}v$0rq0-1nq(}0RVS%0S&zZ}2pBa@Six0]9a:>"
    "@@-D70pQH2%a99vjv!$CjPjBoYCZ+L00rnD000la1oeD70q#)2b7RA65a.g\?00ii5@S8]#@@Ao6@Sro40ehwsjPLJY1m=A(00930000000000000933(WO)v"
    "0r-r6@@r300000000000000000hL/&jp8h>&ckw91oerb1n@Ab2(jz#1owMtNb{eQ0R(-+1owe%@@h@%0R(Y}@@z={%kQ.)jP0ef^$dG0@S8V$0SIY*1{1V6"
    "2kFDe0QfW*0rrrg4g9vJ5=@gq0TeJ6%nrl50a><1jojHC/z2sC0rh/>0rrf20rAi3%nA2[E[7<)2Lwb[@@&r5000000rAf2@@Jf3005$WjPt@-9QA!a008&-"
    "0rJr31PFM50rrb)1Peo000091%aa}F8tByn00961%nJ920g=.Hj]+GZ[^bo@000i50rrf4000000ri60F3PNU1o5D4@@A60000000ri3000030%1U$}i#wWx"
    "ujt/V00iF]0TnDd+M!HU@rSO*6zzs)4e*m)c97\?0E*=C-3hj5){S58AUxFh1jI8X^e6!s&{Yo2M2jQ^O9S#VG}Vc<o2j}<80q##(@tdc}Rui4!0see}}$eU0"
    ">gkF#joWeG>.YAv@S8!I@%5e)0S@G8@S90ij+2v#fb7Ym&TkI^}t*Z2@RDJ]008&\?@Rv5<jk/SDu\?jm1aP.j+7u^az5C(Fq0R#Z7W*.V%>)oJv-ovc-76:)u"
    "1OZFz3jbzw)69f2jB[d:j!%)@4f42k]xL{j^ffmM].XQ^sLBwM3h2br})dmkfyPzE0pnCv1o[J10f1cujpIe&pX$7]0rJ67@R(D+@}={1@Sr3o3iP@40rrc1"
    "><!us@SAo60poK-0UfX{5]b24jOwZp*1uo8%nJxq@q=->}V3M}0000c1{s/70@&l30AT+dd\?8sR00Ac10SJc107-3bjPLY=3Jp29008]#009910000000004"
    "0rrc2000000d+^P0SJf200i91000320ufxwj]tD.)Hoi$1onr40@qY(0%eP80@J620SS2$0rAi2%k@#p<mrxN02CUo@@EVJ@Tu1\?jo+cC4\?ooA4/vx0{YQN9"
    "3I%/31nq^gDiJK2{wd2X\?#L870sn@e}#u-$0Sr-v5X-EQjok-*uL@!z0SR)90r8]#%nJ5#0rre[}#M/}@@Ao7@c}Ja)!u!-0tCOn@@Ac3@ta2Lja@8buvI$["
    "@@K(V7#0K(@T/Fv4fcm]1q7q55FtiO/#(nY]!l)Z@S\?P{3J[e{!]lv&jH-XbE$Oma@S0A90SI#[2NgCh0rrlq3hKG30rSe}@1n><0snY90rz#$0@&Gh1nF^n"
    "jpp$&FXhl=0rz-:1[Ro*1PXS80rA04bRQ#m0TY[W7x$lK1{1=80S.-(}V3*11v/:!jr$C!0zJOo@@zY@@{Y:T<O]kT0rh*23)L]=00-P4}s+l<3I<w@1Q1q@"
    "1{>>U7b<*9jotcZ>\?b}m00rAd0%G2h[C97!@qlS[O{}1#3&Iv74*@:I0rJu50rh@#0rif91pGBmjO/1T{2MCP0@@Cm008D]2(:-a1oF.)XpK2H6bZ#.NGr6Z"
    "}V^f41od#{%nq(@0e%uHjp.GSlK12p0S@Sb1P/5d4grdm0Tn(v5Q$5G}2HQ9{7VY{}#W02}u^uc1owk\?[(36jjP8+G(jBd00Sz/H}${S>1o5G61pB(syu(W8"
    "4iHdOg6W<.1Pel32lTn&0q#%1%gnTvjEjo1OrKD/0p*zB)d6&w}uW673h$5x8uYA$3<lFr238#ylgv!V}Y<=3{ZCp0(CZ&Mjru9L5=z.g@@rAr0qDP#}V3S{"
    "0rJk$3&*Ii0%eJ7%a9wP6y#Ce0se&f0S&k$@$h.MjoMl32Omdr@}#0M0R(G.2Npqb@R^2*e\?}{*)^HeEndkg}@r0020qt*Y1oxbe2kNKPjpr7OFarmz2MbgN"
    "1ow/069egi0q#iKhRWgi].vL*5d!Nl)!M>U}#u^4%lPp5)b5>OjocPijjafm@@St]0qVT40rri6@S06aFJzc<%lxC27w^!a@q=(#%nR#%%nr010sujWjpi$2"
    "!bwRu{Z^OW0%!2<4gJmf1]n#14s=[\?6:d)L!=+)S*Xy{I}2}Vn@%O5*[*4rRjPUA^@Sr910rr01%nS92}VlY%@@r5%@@Sr7@@r310wG960rq@%00i9100931"
    "08{^bjPCG)1@WBw000f20SSi1@@r60@@r670%5x5009300n@Ca<P22Q00ri3000950.*zVjotG(an=9t0rij&0r0321][M40rJt}F.Y2w]:\?lU5D.\?l00000"
    "%nS9000i5#%7^V:jPa)z2pfxK0S.Y$0S-#o@qDV%0%O&A0rSe@009o4]zO*\?[b-Z(0@S610!o*Q9{ob%jB[9}aeYht0%5D/}$H6*}tyo(%nq{d2mgzi0SR]}"
    "1PwGk3<cLh0T[&a0SR@<}BjE.jp\?w/[DwQ*{Yf-k5Ia4>{Z3<0%nJ9o7Y<s70rh&{4^B01}#cTa@S.{H4fVza@9=rkjwN[!L/-Z-@r8e14-VM(00ib%1ow=C"
    "{#&dP{X>Y}]wZJd<O/I!02$wd1P5900Pc6:jPLcw@nW\?T0rq()0SJf300001%nS65Vvz/{2N7In}Vu<00rJi3008@$@@Jf10in+pjPC//4Gcnb008]%00031"
    "@@r300002%FFN\?V@S8{0000000000000000000310r1&AjqdPxFI)Pf0ri{u@Q6T*@RV{0@@h{l008Y)1onr8)MQ\?p+MDw$%mMS@0Sz=X]^FOhi%mzrO<@>%"
    "@})]CbooqH6y=Os]WJT/}}*uV5bk=6rVfT:2j\?uT}@uc-d1On8eYXP7jOKB5)mOb@2lL30}RgAt4/DXj@q(J+&n$ZC[=P)k6=M]v5cRFa@0if3@p{u$0Lr%i"
    "jPs*<lKA![0rqY-0@Se#@q(@$0rr39F+N}4>kA{7<]$}<1onD700i600r8&@%8Bn1jB[mEf-7}L008%c0q#@[%nJ61000030S&l10SSr5@X$952)gba0r9f6"
    "000c81M!=!jOY:5+P3J+0SJxB0{PNM}UZu>0@R#}6AOHj0%wM4}g2n6w!2BS1mui>@@0942kTD[jP+DN1#MzY%nR#)@@Sl30SSc0009360snX]0r&k}FDbYK"
    "@R(&$0r8(%0rr90%3{tBjlV.b0R+*W1P53zFOj/>oagp$1pa.Y!+{AX[D5Xm6:A{@}tQy8%m/N^gY^Z-@Tmb<joDucbOHaT@R#D60SJ8%0@&l0%nhY!1POD5"
    "0rro71a^J8>>UbT00Ao50rrc406P8fjMVV6cvVg8@qWin1nqJ*@09l4@Sic63if]90rq%4%gE*&^&d/S%nhS[@@Au92h3)BjE{(NE]}%f1o!Se3glmC1P!D4"
    "1oOPb008]}0Tel3@QfHK1o!#70rS2{@@9f80p]BUjp.7!:w0fs2Nw(G[DFL7)E75/}tQT47W(}JinSdnvHAOG{5>f>0$LFj0q%1X&xf+BjPLP:{7mo8008(}"
    "00000000000002%000010000005o3i)G0j^00000000310e2sWjoK^a2Q]VL008@=00&M9@q=%0@@Jf1!-=O\?1l*rEFDt#U@S9310SSo400Ac00:01Jj4+21"
    "TwO&}0T3-lafK%Y3Nt-E1}p8OdD\?=A}1Ymx)=<D.[/4c8}#X93}VFq>9{sBDjP>xV4+}3-008@[00JMa@@h@#00i9a{tt=K%n8:(FDt@L@R#]#0S-o3@@A61"
    "0h]o5jP>DX(IHO.0rii40rSSe@@Ac10rAfb{Y7*8}#V($5b=If@qV(#00Ax70rW]x[pfvmjN#i43{juq}${*12NQE{4*SBr0%5Y468r7u1O-z>0@\?x<1]!V4"
    "1Pw&h0SA5{@6&a>jO4zbF=kcx}2g{m4ELk9@p*f$}#W0ca][OC762AQ0WW%L02CIe@%fCA@q#s73vVdljB[mMdlN[r@@I%3}#D)2@@r630SJi1{X$x[00il6"
    "0rzx%9%@bJ0006300io90OBFujPVhZ{-/+w0SAf20SAc21owG60ri35}2*&$00JA8F=CPG}#D/$0r8&@@@z@{@z+6djOY6U<STW8@@ro8@@Ab#@@r62%nJcb"
    "0%ex30@&l20POUE^f6pD0SJr7009c30!.y\?jocI=\?pzSl}#vn@1Qtt6@Ruo<0S-e:Sh{9Z6z(M{(>6K91oFP600if5@r8($@&4m^jPLD!1mDi<008$+00931"
    "@S8%00SJh%&^Z!E@@&C]}VcY$@@r61009300092#09In.jP>#*8#xWp0rA7O00h]}009620098]6z8>h3JkJ7@Sr90009910ri300095#%lj[}jPcq^Rt6oY"
    "2k]x}]8F1D1O@Da@qcx&E>aU2]xV.L!ERVD2N*Oi0RD6&4GLtQ@u1r/jLo)e1+v9R@Sz{v%mDM\?1]d]})!#!M4hFvp2kwo0@ISNEl(A&y2n4sA0T!@80o}!H"
    "jphM3T)=k&0rAJn000fb@quS}@@Ae(@@Su6008##1C4NY5cq!h0rAf3008#%@@SIRjokf^<rBW9009eT00Ji100i910SAc2]cKbB{ZY]D)/&Q\?@@r610@&l2"
    "0SJf30f:@)jPlUXkkLxX@q(J\?0rJPj1n@e#0S-b*H4fZP<}f0(}#cV}0rJf30S@o200Arb2c^b^jNiE]FKPXA0R#{q1mj8d(<rK/0SSq@(jjM[}uNDgh0kz!"
    ">(I7!01YOB0rT@sb(iFfjPkx.!<YSu%nJl700rra0S-i00SAh(5:xz{@@L(B0SJ2$0SJf3@@h]#0096102nk3jOQC[9E)[T@@9Ja0}#6^1o!/8@q3v01P!O$"
    "0qVG\?{YZu[[CSm/1nz/@0@r0b6Z6]Nj]C=)7Z6)t0rrcz%nz(}%nq]$%nS8{1oFM6009300!zrF}t*J{0r@u2%nJf41hddVjoch@7XN\?d0rix9000000SJc1"
    "@@Ji0SEK1o}#cD:0@@k@0SSi20rr2#@@r620D[f)jq4kY(iNE}0%FTP6969J}3Nc00o%>5Wa+M(>LK&*<P>z/@}MS$@R(V@000xk2!I=Kj+i.edEc<j0rrl8"
    "009f3%nJ9100004@@h@#00001%nA35@S930000630ri610w![mjpz}&-V$=:1P[5S1nzfU@Si2{1]XGh8%z^G^JuEQ0%]5g1PwD9@q(V{0ri3500q)jjPMo5"
    "dgMvU009u70rAl40rr9100004FZJoB}26^V0S-l40rr910rr2#009600g[pxjo<)^*Z9Vy%nR&]@RcA)3MwQu}VV<bFK**X2k^)3\?V9fN}VD/}00rMe@@Sx4"
    "1TNQ.j+jvElic&Z0rrll0r8&%@SAi400rl5@@Jc10SJi40{Y)P}UZA{0r993000620wKY}jMCOdW.9st2NH@&0l>fjlH+l:4/#T-02usv[^Uvj1TH5JYDczt"
    "1[@Dj}ulcD[p!vQjPT:S!e[R&008&(0S-x60rAc10ri37F2T<W3&7b2}#M&$00001@@A2#000320D7L@jp{Yn1+lar}#W9f4f=F8[=XT:00Jrb}uWGk}2*{h"
    "{ZdMv0%5P5@}39*1Qknh1k5A1jOem4FX}A=0riQb}Ugc]}tQx]0@-cd3&Y#41][x6{Anu+Ee(Ce0qN38}u3D:@6%zoj]CV//AqcN00033%nJc000ri40ri5%"
    "0rAc10rrc20S8*b0s5S90d+dJ008]@0r/3XjO*qT\?u:F(00hIU1m=i}}UQE3@QZG{FG>l(9p:j0>)W:{0%wS70Td&>@q#&}0HLmpjO/A-01X=9@q=$[1}gt1"
    "0q#&@@Rur#E<+c(@Sq=S@1oky@@r5$}Vvc3@@z/\?}4JTtjpzJk!Xzp]@@r8)1PdSU2m7zc{5>G}2NB7M/t3Ie6B<Wq}2Qu)1POk#0SJh$0n0OujPT*-{aVHE"
    "@@Ak>00Al1}VcY$@@rh@lv*67%m2QR}#N350rAc10rr9100ic20cLL)jo[pYGztSG@@9JK4clsB@}=-@%nz]<3K3ti0S&A61(]Sj0n4hK%e>}@@S7Za>>2K8"
    "jobM:E*(+:0SSc42MK&e%nJ92@@-DripD\?k}UMoI0@J300SSl40r9c4}VESx4ka$Hjqxa.[pyin2kYqgd/*s!0SI]#2kwF^(<Tsl]9tf%3O.kx(js9#0uz^N"
    "}Vv682l]A!jqx4AE)KzA1QaFL%mkmC}rdXZ@%5Vh1P!h}2M+!yaQ!kG0Ub5j0%n(h1{sqT5(y%QjB#Vcl(CN(0%e6d%nz(#0S&r4@@r8(%nJf30SSf00S&h{"
    "@R#@#@@So400ri10txy7jB[gccne*!@RW3k2k[&j0@A2$00ib!@ril0@@r01]56uz1n.][0Sq@@0Si9a1^aUmjo$Z7fO6u7@@irg%nIP+1nJ5#@R#Z24ETG2"
    "1Pnic]9>%0{Y70!}ju)J1Pexi3Irj3jrur-Y:1Sq}u3jy}0m\?D3Jx#d00rqU3hK=32M+Cl]ABMi7e>ki0d1eq1owPi4v}iQjoC/&4GcCf0SSMl@q+]f*W1\?d"
    "@SAk^\?sHTB00q(>2yO0q2lUef00B8r@S8Y(}V9oPjN@NCWb.b!}#VY}@0rh#1POP7@Sh#)0Sioh0q#)01nRD&2M>29@0rPd1aPO-g[4hwj[/Vz34ScM0rq(}"
    "%nS8#@@i01@R()5@SSo40q#{21oXM81{1J2@@-o30ri941k}fljPCP^*4tsA0rrgL00rl6%nJ600ri64)=<Du}tsXW@qV/$0000000000009310bqT!job@)"
    "]7q!Y0rh)t]ml]p{zi=b1n&q@0TQ7J0089&0%nM1{YQ)2@{.6M9@#JK18{Z}jL{G]])W5I@R^3W1Nt)T[CSy&0Sq#%2Nf(00%5o91sK<7df}Zi0swG000z]<"
    "{-P[QjPLVH6W)9.0riu3009c10SSl300i94*KF@A}Wz)2@q(]#00930009300000106x40jpouQbP)0/0%wn%00irc[bSB<1n&e!)M%A)^8#^.E*=[O}2ZZ1"
    "1n@5#@@J31%a<q8jB[Odd}5CU0006e@q==}%nS900002]0@@u30rrc00r&Pe0SS8#@@Sl30rAi203FIPjppo\?-w)$#@@ron}#)c41]Ox2009b(1Pwu50rrf0"
    "0y8b]+Op6S@@J900rrf0%kdx{jphfK{Wed!1oPC!}tqSx00h@]@@SC/0FfsO].xCj2M1Gb%n8&{@@&Aa0S&u5@>A+6jOeKQFN!SU}uEi301!L%{5L3&%nJh}"
    "01xeg@S0920rADc]-b0}1[\?P&0*k5Ug-kqCjPtxN*qJ%>@@A2%00rr71oeA4%nS6ggE}-B%nCrl2ls/9@R#(#%nJf30ri3004C&ZjoKu&3!+^=1o/Uh0R)05"
    "2M$be0rrurqwSV72\?1oD%nI&(}tZG}@%e2[@R()40)PQHjPUfO}2pr]0092]009l30099100030E*MX+@q=6^1P[=a00000008##@@r620C@JWjsQWA1d]I}"
    "1PFcc3\?<!]3KM.h@@-l43I%=40@-l0@EGy!w9:G#0TF@h%nrff2f!nyjOnBFS=XvD0r<kh}#^30@r8&@0%5Dz@@rr1@@h#}<Q-11)f9Q)0}=o\?0Fen4rq)Dg"
    "joB:=9s-eJ0SA92000f41P!M500935uAu&V3i#b31owA01PXP6@Si30%nJc60{@V<jO+:01apsJ0R#xs@%f5j}#cY@@qcHjRgf!({#hH>9SMxy}t{M{0Te2("
    "@})Ap2MXhzjNiHB[Vz@]@@J3g0Sz3Z0%eA300i940ri04@S8]%@MP&-d^d6s0qMc-00An}%g+8=jo=2@}#^01009jN%nI((@Srr6@}#^3+Cr>i]xN!/@r-Yc"
    "0SSl0@@r000rrc70[iffjPLAO{VRUT0009200992@@A910ri34fN<ux00Aub@R#{000000000000000005+5ojB)Uc69!Bh1oOGx2(Bk)3>rmt0SSe%{4!)("
    "@qV](@$-(S&qJ@X1o!(n%nJ2@%44K9jPkpt3MY6C@00vR]-+)4@qDM{1PF=qsoGxx}XWJg4*-jo0SSx80rr62@qV{31svFGj+i.o1oFS7@@rfq008]}0rr90"
    "000351onA50ri6100iof2k[G30rh@$000900d&yNjPLS**4UNE@@A90@@Ac3009610ri31}u3<2@@r2%F=t-L00000000310095#0lHGJZPEt$ZR[jkY/80Y"
    "j{\?xYjT*l7Zfx4QYR}n.jQhyMjPWX-Zmb8!Zg9^TZw%OMZo%*xkiOKWj[\?%pZg30KZsuc+Y>f0mjR0][jLFGgZc{tgjs1Z3ZT5wuZ6!=rZp5ETY!7PIZDh%."
    "j+TZ6ZmQ/NkdkvvZn}zFZoID:ZnDfGZt[x\?ZFg$+ZjbKtjTOGvZr}IQY}srPZadI#j.@SBjY0rkj@2qxZ9xQ:YY#nyZCZbPY(g)8Y>$^&ZLo-jZbs1yZML{7"
    "k3a7zZVfA^Zp7InZcpV<ZsI=kZv[H.Z7H*tZ7RO4ZIk1CZpg7lZaCE6ZVi$WZu:gfZcF7)ZNMH9ZieLPj>U6eZwB]tj@a>/ZFVAgZa7=zZC]@JjzeD6Zg=/C"
    "jZsd]ZVQ9HZHdP(Zonw+ZkvrsY^H*tZW4kyZvT!OZu\?1ikiqz!ZE-wMZSy[Cj}ZS/Y]nn#Ze\?$4ZHUI4Y!F$ikH]YzZQP[NZ9yZ/Zv&O%Zpz:}ZoaH@ZS#{R"
    "YA\?QnjTI/#k3ApVZK}5(ZnM3IZdJCFZKWD/jE*h.Y/q}}ZJVW@ZI$FVjY7J:k8@:jkd.</kg@CkkbQ)#j:g[qY:fl8ZCu^&kcjf*ZhNzaZKhC<k785*j@Nn9"
    "ZdgmkY*rpWYw}(vZrf>gZsA33ZMfDBZG!g&j#wlyZX2DvZ76xZZWRAPY:lFMZDk=rZj)[rj-^V-ZC4ppZ9RONk4G2oZ8A]{Z9}H\?ZeR%^ZLxqhZbL[#Z3dP#"
    "Y[pwyj<PozZGo0QZsZm>Zhi.*ZeE-iZNy8vY<$e)jYMj=Z9b&GZ5[$ikcF1ij&U-EjJwt+@iADS1oG2lSQxib+*Ni%Uj&wo8v>pn^7ZaK9ML\?#[F7&)1ow$Y"
    "3*FY91pc&Q:mn:C3j4*q[Y&&U4h5s310v%KhS8{X2nMt6%18!9>E7SD>S61u7#9UE56+1>>\?EuB[.<f25By}p^FiFo*+yu3\?4oz>=/KlV1>9JaF.n#L2>2[J"
    "\?2Wdd(<8FC[:#IX4=ts-5Kr3.cWL6gjok-u0$knn[9FW80!HdU3IK]b1PlKv2Ny}o}$ZrX7v\?Uo-uA.4@2k[9}viU[(ga#K2(j5:{V-h6@2+[30ora@7x<ah"
    "2lT.%<OoqS(H+z]{7!qU2LtC45GyMB3Hw0E81W\?f]Aas%)!^+l0r@L{[cn[a[dbr{}XZ<Y}s\?>^5eC9S}t6*M1}gXp5AA0G3i{Fp(g:lg5d!gD@udhPjL*ez"
    "@on1X3Med7]6lCV(g1o){#}}k{#q-g>OjZk@oadW[b:3K)J}(d}0!>e}7-cC2(s.U\?V9J0}2g0@}qZ]x5F+uB@]8l0104)t4liXR{V8S9{XBjV]7jEk)FMeI"
    "*ZBbH>LC2/3i\?7r)<N-u[d#AB\?miqm9T7Tc1OIYO}reaV5}HL]1q\?TF0@./}[a>rl6z}v)1tYx[[j{Rhj[xuy1n}S&1POz$0r9963ibON0r909@RlY&@qM*1"
    "0RV<80%5P8@UkIj2lC2f0Up}8%nrl11n&u5@SAf60SJf4]b)#i2[M.c0qcr]@rIM&@se]835!]!0%F&h}uEMh@S08(9@(xo%nJ]l2M+8d@0A2{}V3&$0rr9k"
    "1Pnr3}%Uc[}VV{50%w=h5bVta0@-J80r9ug@%]5v@}v%njq3]([z<cN7A&#11l]^<0}co[]1jY91P/k=9@21o[:eG/1PPs!0{]{Q}}8>!3OJE&2&$^/>DsV-"
    "&sLw<]-u3k}uNS0e8Rfx1]u<d11ABr0(->p9T\?k*1muQj.Sp4:5eB-X*zGO:6:20e0LOTs)8FS6{YHfN}U/)!a]ZP[0qW/m{uijUVMxE70]&}](g+tZ[e.bK"
    "%ky-70ly+.jN8\?e-q9:z1Cb!s>kLIF1[@ud2)*I12j-Vs2icf&]yhON6-Xds*X*<x2i#Wm2k/hj1P]kb0R2*w].ws^@}D<7@TNJ.^H2Tw4bR=<@0JVb)d*p#"
    "0$C!J8uwX4}$o^:]zvOy]$P!#1OA334byN^@Q5}M@S8{b}tQx>[+-gM@}MP!@R<$x]z[L41Q>d}@sYqf2N6$-5d]iH}.GQZjO!Wl]cpNH8p^/<1T.F2Zvh:j"
    "}T+vt@nELu67R1<}SmtP4fcIb/b8{m%nS2g1)^09ea&&o0S:bD7wOKo}V3A\?5d6xI&Vei{)G<Wm9tnbr)Gr7W01E-:]ZII+{1w{te9)9o[FycLZx>b/}pcLF"
    "3Mw\?(@P1gR5FLuF]AL&]2&UKt3&XU<71q)U@m\?m}=ggJd)fJKA7V:2&3M>*/jN&pf}XVHf0S<RR2]f#I}r[)X4jBsC}%sBY[BrW-*3Oa[@PrtR7.uYL3g)=="
    "}UISr@p{ab)EqE*]-%vz2]8y!76@^<1QCL)5:e=(%nhpg<O[Jy[=]<y(*%=X87C4%e:$e8]#btU1Op.A0p7M8]-VbW}ulG^9VI]^4/2XT]+NY7@RdnI<BLFi"
    "{v5az4myi+[cw:oL)Wo28Vq@+jPU&^@@&65%mlo<00SYf%nAc13Ku871]Ol2%m=P){4!K\?}tZS%}Vhoy%mVWa@@ic6@Sro60r-ua@@SPb1m#G[1Q1S3@%F#i"
    "3h:(h0SS8$}tyu[2MB&20TGbr@pfWZ@@}lV{w$o(0Sr023iQh81P!z@0quJ>}1Up0@S8%11POJ30%F/30@I##1][G6{2MhN@r8&\?@qM-@0rIH4jPV8K{7=&Z"
    "{5U0=[bJQ<}$Y<&2L[M400rzT>]6<2776s8[=Y6^%nrb%2N7Cf0NVFL{5+a03jcN#%l{M<1n&h>1iZ#H3)trx@@RT9[+<3]@Psy%]ZzeO3\?KGb5apbf@q=@-"
    "*xvrz0S0Jh@SrD8)f0s)>(I7\?[=Pgd0S030{5aKU(Kjab4*8Xf*w\?uN}$xN^1{2ed1RQR6]v9[[jPbD.*6cl{[+&!d6-E>n66PJi6Dv}/92Ot\?3&Ze14!Y<:"
    "4e/F]b<$0s@Ok/OFE8[C>hMHKf+B-V{8Kg)66fg@5bbh8)H/D$]8fZf(fnq[{v=VO]8e7I)(y7wm-#5g@wt2W5b(\?&0u-y75Yrk=3C$S[<]bUQ1NM/@Js-8d"
    ":QOve0TX>=>(X#z3I<Yc5+kV\?)I2<d&SY+%}g%E<jOYo-[+:N0]AR=30tk8i@@I%61n&t=2(>zx6:11f}umJw2(T![1]5b/{xzcn0@8-={Zv-d@1G2m1Q+Cm"
    "0qE3l{w$WdFEQ=g1pLzw2>!w){XUB3@qWc40%F&g3&Hd#>MIa>@S.({}U6>4}YL>k3F+jE}t7l)@}VP$1qI7p2M+eg1H!sm2NHUP0SS94{6Ili1PwS9@qDN4"
    "0996RjO!aE2&r[[3CMxf]AbWd+\?9Qv*7K[v[z:#*g6c}i}vSc15cSdf4*Jdu@Puf=[+Tl):O$}}(ad]c=!2=b1Lr>I5DqX2=6N@4{$#7>[J=<h0tB${!=#NC"
    "6RDnB)evg-!B}q/2(Ubh}$Z9)/mA/2}#k*X!Du{c9S1%@5aYns8w0OO[rrJr.#)/+4/b/4)\?1E62nm<j1Q9HcbGbqpjpzCX&Q>{.].[Kt/acji{u8qi>jF=."
    "]9b:u4HivDao\?e{7yfVO>]yh#}wyv*<{qv[}2Yp75zl@<>K]hJ0Rc0#{66W>fC$F*1OBnK5D\?L1{Z3J=@pZQk1]d%cIu&\?N0sn@O1Qa/20Q])<!9mes\?]=@y"
    "9tnbT3J]\?}[^i:Z0S-i01]!SKHZ<Gq0SIM-5c\?hz0WWKs0#@pl(oYp6jO]9H0sf.E6+7u[1p1@a[=xv1F+w:j1P3ao@P%N&O+i6.@d>*=!^7gf@]qOW@p*:0"
    "67uh90%E].1]v(\?@roH/5csoZ0tkyZ4FoY5=\?mh50%X+@@%5667Ya1p2L[h^)FmjZ}u=0UXEwzq1Qk5j4epbj2(B=i4g&Hq0TF&c0@@2X1Qa2L1ModO}V^qU"
    "{zs5s]8FvH/9yNw@3TZ}joT^v{#Y%0)Dz{]^d^kh[DeK%1[I(D2hnUM4H}\?O6::%F}V=P@}ooJN/-iq\?{$%=!*Y4cN@qb^=0t%95]yz[()F3XA0Tex30s5q<"
    "01P29}owHS].5H\?2KIV<3ju.j{4!L2]z/431@nlW}HF[I]5wr#@$bs!}1K*/8vV@j75dC22lt5J0@zY[3<3ku@q=3-01pjr3IsbQcW!8qjPtDT1(}Dl]917R"
    "@p{P}<]#851{>w4}SN\?+75E[m4!}1y)f9E^0Qf:J}2p@&3\?>5b]9b=41]wGi0}V)l0sGCp1RI40!a}+$}2Z=()/SK:@o<Z>}#lY%@r&Se694.e3K3Rx]-aW@"
    "1p=1u[{}5C@[Moj%l{x+2)pek0UC$b4e*Ih{tlb-3IKV501P8l]7Av\?1PnG31{j(73t9#tjqmYl@{Q/#{X:KU[aVFT}TAnB}tHG>3j3[E1)b)%5zV).@TF\?)"
    "@o$f*(Jd[d[}3Yk]AUB4%k:aK@}D/F]:pD#%jN8E}4i#i1oe!U{Sj]&@5B=^}%<*!1n@5!@}tN+@%nIQ}u3P%00{{H]ACf@(fFcF}2QfT<mq%t4Gt}T@po)<"
    "3h$5E]AkI]2L[A3}ucM[0TZmT2gK4H41eerjOPxU0S}V:0]s:R3k8Id}0F1T\?5>$^>mw*x[gCZ:6Co:D7Z*GrW\?f5u4FzT>!^z/j{u*&X}uv0v[/ojR{4c@U"
    "}T%pY\?wv3*2(tUu>KfeM}#2jz}U*Te1oPed(IIm]&uGr$0Ow0$}zOG!5\?-[W{V6&@2kd=+92XCw]y\?wPaLWkH4egzt&}@p$3lX^faQuv5@@96l3hqYI(Vp53"
    "jNz>g6A/u{4s[=:@H:9b@}M:0%npvG1Oi3d2M<(a)dIj{68qOf(<A*>%m=rK}tHT40S@Me1oOL@2<zu[4epy@5^x9I\?V@=#<O5V/{w1QG@qW61}Vv&L3hTVc"
    "2laM21]w=b@1xUf/-Ill2mp){3<usi1{s-e8vz-q1pB>}[b-4Q0SggA0@}&{}o/lc}2pN3]By*62lUIA6GxY<jNSgo(J).cbl!F)4d@un{XCWe<o#p#0pXzv"
    "5Aii:PYVgH7SVgQcjDgi@S-nZ0Q*=#0}U1q}tzY44:>KT8W1gH6BC0A3jEa!2lkeB00Ah>1]Fb}[^t{v77G:c&q@SE6Z=$m{ad/@&s2!w7u#H-6.RJ]3IKV2"
    "6-JT^7YTyw3Jyd\?0$ak%0n#-O}}7-0{O[8[TT0t*^/Gjk@q^68jP$MQ1]]qo{jikw[+rK)<n6b+{ueQI1nS040@A08\?$3<H*@YM[eDi&)@Qy0{1nAh#75XZx"
    "0@\?/w1o!=c@@&l21{z:z]9tQj]-jTU{6}l9{XBT(@SA(p0T5rf]z[T*4/cwf@@98)0TY5h4^@S.1]4rH0R==*3&p}x0qMV^1{Cq80TW]U}v0h[0UtLh1Q+Lq"
    "}2ZAt{vFNT0jiP*jPaT^}ub^E0S\?xT@R3E13&2nQ4eo@20TO${@{Y%#0H0M+]9kc]@kX)8(\?Q})@mOjU0U+Xr}XPLy)!lRx}3l-(-j2vl7z3Mv1QUIm@X9Qp"
    "}X6}I@E$dR1{1@k=&-+u@%5=j7xTml00-Y73&75l]aH{b{8JSt)/SQ}}${0Y%mC)V}1+9%&rGe*4!*220S@Pf0RM^e2M<]o]3vp7jPCDO01n#20U+4u5=q^{"
    ")F)K>cHIrd0%FPb}#Mc\?)!uL*)FWy/@q3x(0]jg^@pZH1@}VEb[+&q5}T1Q>1m=K52(B=83k0gp3I:.!@p{J<1QB(d2M>.DZX4Zb@%O]N4d:.(1mMu.[Ci7>"
    "00@P5{679:1]ere2[c$d]:7P@0@8P=3IJG&1NVG&2j}>C1RQ[x1p1=6.t<61@(SqLjP@>q@oi8j}#DP>{v!vZ}#(%2@SAt$0$Cn81Per53JY@8@rrq&}HF>z"
    "68&4p0rI3X}1KQz0r8]{}#v64@@S60@/)P=0RMx/0{PQT}sjpS}X)@Y@RMV}1[RP*}#c<20TnVp0Sq%11o[h>@p6^\?@nW}.%lfvS]8OT>0S8PO@0hP]00Aob"
    "6AOgp<obeT}Vb{/0r8xY1{T-d1!saTjO]cY3&+!O1)N/u\?Vi/N@Sh0n!+Y\?7]]]!uh1}pQ3f.DE[D/P@[*:^<1)MZ1}wPm(>MhdT&(hAm}34FY@00Yg]Hw&R"
    "1{r-u1PFw1{CR{8%mkgN]$GLK}sQ6ZcImPQ6-XvO]Z7Sr8sN$q8P-l=(r&-UWEj[e0S8])1O&5xlGw^R]:pTc=<Ze}aQG1@[CS4K2j\?]=1R/G6{7ibOjOX*K"
    "{5Cyk65ADL0r\?u3@QQD}+lwdy@QGZ+4FGw=)Fu.U>N4s&0r*NC2j-e)0}^354ee]!(Jc$$2mgLd0%OA22M%F%)gGu#3Jyqf1Qs(q}t*M(00&&CbkIiD1p+vS"
    "8PTf3{ncBCFD=hM@Tohb1mU%K2ls-c6Cy/E}tH6<@@&k:%j(S20pOFs}tf%L]ah3c/8LKf@qV@F6i{fnjq]i.2<JGg1OF@7=&9RA(>c(q(*aAJ49[**4i2v8"
    "4pfXH]+dfn{CrP-@p7V!>)ic}[fEb![bh}B2Nm=\?}v-ut!$8BG1o!-40wi(({wbQE0rSPu2DvL22>PkcFE@O0(LfBm6:Byz1QLw5@p*p5*wziX2NyCl}UQl/"
    "0s58N1p>qt2kE)i/-zYIMhg^+[bycz@P#re0O(.^3B!A)jpzV8>}K4-0@Sib)GrZ*1RQXn}49/j1SdHS1{jz(0sKwV}#V^80r@Jb0rhT44*hOi5!i<[3K#!w"
    ")&rRH2l2.Y2MBM83Kt$[]:H:g].n7#rr9149p<dj3&xtP5!>fr5^PrX}Vmb%@qV]$/6<7waPibM!!XI@/Be2N>a\?+Y2Om<Z2m}B%}1CE52Nyk3}VCWX[b%12"
    "0$3QV3}pf#j]23J3KM[E6x]FX0$K(2}#v690@J8&0R#-%0RJt]Z3(mc8Vx{I+(SE{)DB&L@Rc^31nR&$}2HW22&+^W3>&ja3<De33KuFI}2HA$5=&1b%nqS@"
    "}udSw]-ti]0pG:%}uml8}2p6![cod53igt72K&5*%m#=*83PWf5D-ao@@Ah}00ro10[D/w@]%BR[7[0A[EtuY<{IsZ2Io^VjMtP00o<sH)f7D6{4wE=00JVt"
    ">(7[:[67!u0Ten}3f.G>}v0A9}4B81{wCf{1p+>a}1t98(GwGS[:t/H@Q*9&@tu1c0o7&\?2(%]3}4Sw)}%J$!@swYF%mDG]}uM&@2N*a]{X>421}Q[c1!(k9"
    ">LBJF&O>ms}2ZG*0rJR-@Ub.o34.Q.]8X:+{Y7N8@qDNc@}#u+3<aP!%av3fjOeEx0@Su5e7$Em2!OT\?[+Ay=}$EF\?0RMc:{XjK/4FG$$@RkW.0Ukzk00SVn"
    "1n-t{0S-o0{X$Dw%k<gP4GVIe4G(>l}2QQB)!M\?M2m7te{Y.k<0%n&B1][R}]Z\?}V{52d00sFD4}$Q3]1Q$+h@}kQ86AnaX01jwM(h*>D)*]Q6}2Ql*3hi(]"
    "@Sie\?^J5n>4e*28@1{jF2!&#^jOwyx1R*UH^Ar^d6w{B@)Fmy!&}7Cx3gvDF<no5-&TkzM@@Rj6]:pf^6:aHw@@&t%67bz{2{.7)1]oki3!7.01QU>G5c8zJ"
    "=GK%p[C@T)3LSl[{Ynku4eRm:1@^vPa[2Zx)c1lu(//b[8@*vr9t!eP}%>y1(&3O-/9I@V[+hUJ@T5FX1HdX90Qx{R6b7#S@SI6K}]G0A}rgk#jP2fM0%eD1"
    "7khFG1b2>C{w$-}5=FG$}toZ!1)V9]0R=r=[c5y^6a9B(2>4GQ[+&!11OIJ<]-2G#2j8r)}tH9U6:kuS4His@4eG@5/6&hk0r@r50SAL#%nrDk%mb::1oOx2"
    "@quG>0Q{2W1m2Jf@{o^P1owxo4fEHT%nr6c@quAY@@i6j0{GBR@S-+[}3uP)[=/l]@s6qk@+O4-jollA}$yD)}tHM{[+-B^%nNVF0S9014FZ1+@000g5bMFc"
    "}ucVJ@VzNr]8G6#1T6A8}tQAR1PXJ10q=iT@p*r%Y/c2&6:Bvg@s!$^{W#(d}T:mX@H%2)@R##(%nID=00-r4V&gai2nmmv)=aDG{z0Ya3L-pk}#V@<2)pyN"
    "0SSPg3KM$y0rr3m0SJod}UZ{q0ranG1O\?JP18}lojNhUo}uVoN].I5O>LUCN%nJxb0TY7#>J@l*[aV.U4D)lM2)gth@TGn3]Bg](]-tQd%mDGy&rYC*2l$n8"
    "1P4@)[c/E01]uEw2>mGR0qur<{3&cn{.Khy%n7:W3JfL\?}#3P{}Up3}1mk<U4&kou{zQ{L@sF@h(h78X}2pM#]6#[D2hf*&{5$^a@@@M92k!Vs2(Kz(1{+M7"
    "@Jy7sjM(g*(cX8E(h6(z>L1DI{wbuV]Z&pw[g$j$@qWh$9{&!H{7dn#}wG\?n{X+Nk4^%bO[p{D0<[/8V(/[Jb0Us&9(Hj&R{yvh%1of2l]c0cf}04tI5ejNy"
    "3hKbS\?u.{u0Rcdt[^2j42k^@^)GJy>)stGE4fnlP]zXLj@0o$G1PHgV[WF=p]Ablq^fxdt0Rv0c2<IG=3gO9#@L(\?NjPbfI]:}uu4Zb&[1m=W1].]c<4E-A."
    "1PFe(@pHd0*4UBqcii(I}3R=K0R3o!@@0f01]eu51{1\?#@RcS<2LXz#1P!I#01f2l8t<Tw<{p}:}u3Y)5c-dv2lsY5@rh<42\?YjO{YfTR[E2r[3I%$)3<usI"
    "1]OS9cKPqp0@&h{}$Zl-2mHj$@QHd6@(+Mc]zFT<&sC>W)H]4822YXejrA<s<&OK=^LyQ#-RPimh5>*2{Ax+Cc>Dm40xQSD7BO}:8X-=82*-.j6z<6B7z+xA"
    "0@I@z\?#R%0<{PYR}%SE#Z1rk/[Ch.09Q\?4<}T2p5&lRF\?>[ACl/UmaU+K=qBi56a&0tupt2kF2H{%]M+]rW%17vpzK+i]&A5bl}e[Ekaieg\?C/]D+Ny@#r0B"
    "0SAVg%hVc2dBe:<<d4@Ejo=8)(jiw13iQa@}UY{)0p]*^(M[1&7xjc#{w<mr3Nb6E6F67U:Qwdl}R}[E5de6@3L8[e1]nSb{45<.(*s-w2[m:B1N2j-e+3:+"
    "0Sh(Y@qcNa2<qm0{7l%94!x#ohUs=V2[u7$}sC3@5C+(875Oi&[:DLZ.U1lf]4<Dx1P6bf686(T0}${=2jR9X}*[U+}##%60p/W{)(CqCjM=pX4hw1On80Lv"
    "7BMVH\?%}MI3iHN(8TX1j0R<$GF:JvPbMe8Q6CnZ%1}I1j0pzfb7XndB7:HQQ9ov}m7yYZhapw5![/E/t].*9P6yugD{.ik}000SE].Oe23\?i()59&&/{%6GD"
    "-24Jbeg{)Z6:+kgfY&GV8u##k4iCiG}UGEq3\?R%91=&l\?{})i-10X6:&#[ahLK[tz]6BD1jPs%c0P%)X[cxQr/2tf2(gbt[5!V2:>gnG4]5y5:i]*FY(<ama"
    "1R7R3>Nd%I&SH7(1ng{C/:Y2a[aNmR8Rq1Z5!tJ=8Zj&6[B)mC]9#hY&U87<<ok[@009lh):z7N6A/AS3>AKo)!)yW&dh0i1mu*q>>t33<ott:\?V-YV2>]1%"
    "{Q6Sv%kiXn@SSM5}VDY#}U7Y@@WOmy)\?SmNjOd$w)HH6h1]v9U)##!@00&&n%j)kI0{[$w8t:Wq<O!(L@{xEH3<uF0]:5H.@q^2{2LOJc}W-L=3k-Hz1]On("
    "@%4S{&S!/{F+O()<QhB-0s5Sj}uk{^@rG}w6!6sU4IOKs@rzYW{w%Z42){%<6aamx}2ysf(gr:o@SA6h@pZ3\?}V4z!1N=9X0quV@1nhV!<sp3z}2y-V6c(Ci"
    "jOOt70QI(M@qVrY2i#G/)h#Jc8X-huaSQ{-5.WA]9@Mb}}v&}O)=>F.(jiR7@RUWD2j@C!bimF/[9\?>82[cd[%mDiP}4AYk1Ccvl\?wtex{U:{c{5UZc{Wd+\?"
    "3/\?6P&0G}(/7w}C%l7{m0M>VJ(<&sZ((2A\?<Qg+y2jh&i0TO&g4g04K4efw[(G[P=}Ti^-1S!^23JZRi38[mRj]+MM}t{A#0}C%N0Te#g[D6443<([k0r@Ad"
    "}%<:OFDt#P<nWc\?1{$nb3I<-y)dQ+O@}Mc-{YoN\?0SSL$)EykP4fmsA@@&P01O8Q04FPF!)GSE^}tpS-0r@e]}rXjhXBPj/}TS$Z2K.x*5BPq{4c^A:]z]gp"
    ")Fm7+(\?ZzX]xV8L[B#L-{vFH-0SzlW3hBG62L^JP3K3v>}Mf<ujNJyp4Ky@V9vY8P1nxCuJ}AO.)>U)$@P%y!a{rJv2>!Mr:&%OixD3Qi2lCF7]At)92(K&8"
    "&}@cf3H^3S5:t\?O1Qt}H&2.wZ1PYs/(<iW#&rPhW2nc.L]yry==h)e^4/4pw{WN}P/A.fE)F){Q)E}ZW3HOlamF2{-(JdyG0R#(-2KpHP%a7)S/Z5[3aQW#s"
    ")-Cg/*#YX6}E.]mjPCYW{Vz.+}UZ)a=Id5e[ydy8{YZ)0@SSx9]yr1>1nR=/0sGg@{Zml0}$Qi*{yNM400q]V0OdLF)f:Ka0qDY]}u=W2{T/fi0Tf5h}uu*8"
    "}rFaZ@0JJv0r&-w(hpU+[ympa%mVP&2k/Rv]49:F[CAH@0S&t]]y.!N2M1tO01w\?(3jMXq)/AJv0r8]}@R#^52km%81qImi1k}i%jMd4t[v3an(>N}D/WR+!"
    "ht4Q#4XTrT&oz-Ql^bzX{+quuXcH/.3*dlt<PEE591iRX/:4J00fNPK&>k*u>kM4))fUk7UNGvN9U)R:!4(yOWgqdgY:Sso[S:1jZu&P>]EHmM&r](Y{^l4["
    "(Em9T^1mO!0rSS+5*]N*0p8MF<P$Iy+O!OO-qOkw5E6fO\?u0z5@2CY59twsh}+R+)j[P9S}s>r=]7@4M&SY!<}t{Y@2K})2@S-ub6-O7q1OhT02l>5k}$Qu="
    "[=674}s$i[]-bkP{WwjP1[@i2]A1%#}UZu{>KX692N*}h}#D%5[a=.R@Tw@l0S8)8@Srxd{6ZD{{5:*!2(j/c{KJwM{t#FV]:7D)@R#V}1oX+[0@@G30nEU)"
    "0$Uzo0T5Db]Akm62mgws1O\?W4}-RGCjq!z782V]b{zCaU75ctP5^W+ja#GI}87t131mb4Z1kK1Z\?#iDGaoR#W0Sli@@VgLy(<heo]zovh%mBLv@})h\?}IC$g"
    ">P/J>4hO&^)st=K.PMZc}VD*6@RD#.{u}>=0r-0e}1Ul@69<u]<R[l[/1q1I]:ZQ4]-j^+5b>[Z1{1Ac8sd(h^!@:p:SgP@{Yfd:1<Cl<2CgY61li-KjP%{q"
    "%lG<+@q^5@{wj)<)/zzC2knf30@qS>1oO=g2ON7r}33<4}S4}U@q^c7@PF*/@0JO)@}DA/}sjZ)@}(=)}mVU90{6mQ0$Cte}SF1Z}tP{/@OWON0R=-@5=SZH"
    "{Z3($(&c\?Y0TGeh%l{c()Ef]L@TPkn%nI#)2l1Yc0r@x71p2nv2)py])FWj\?}$*D@0q#S^2KIVU0zEQ\?jOeEx0%c%Ycv$)^2I+:F].G3!0T)rm1PvY!1ORJ&"
    "+<$-c0SA073>n<8@}=Wa0qm6k5CDOs2lUh87wm.60%3{N4*IXd5!s%Z0Sq{5}W9f71Pm@H00Tbr}t{K0@}uda1Ohc:]8E.H[CJ1w2(Ul<1m#%01]Fob6A^{$"
    "1{kkv]:grX@RVHh2ibB-@Qo^H]A+M%^\?}YC{5CT24m]UDjpgi*1Rx/>4.<T%dM^tjhvfcW3guJP86nk(3EdDB)F]#L5d^BQ2K7mH@(J{(1Q+!H8uyrq4HAvx"
    "=f0B.*Co&(2K9Mi7ZPsr\?xz1Z{0K@(am5H%75)WJ0%n]15a6>@3IC8x6bzLC>Moiw]w\?Z<1o].A!KFA=*9#*H(F{.i4EKSx@@SHS1KDMP{z0e&(kY*c=LN(w"
    "]9t%d!4c5)jKTMc3Nb-J6:M$()&h>$}tp6%f:E$s<QzOjX/zNE3iwf15b=d%3>.jr@%/hb}W<zu7y7lLg6WWfebw[*]8G5*3i/\?/a%=!b)hL4x0#7dK{XcI)"
    "\?V0G(3!:UJ8Ll:9{.jp(*5\?ia2&#<v93%K$]\?2O&{Zm-l53tLR(E(B\?!.mqd^]NjM74q391P[@c>lIj^3loSX*:aZ0c#l%tj[X*H1o[D74dTA31@m1c}tpN5"
    "[bAdR0qVS{1OI=}Yy=m74UsJ^>MQIO0S-n=0qWc31[@l50rJ#b@}cD[1}7k33iHb41QUzmO=eS$[=5E/0S&r5@@@/j1o5D7}VuK47YBNt}#E37>L+k[2ls\?s"
    "3&f\?}0@Se)4e4%F1POP81]XF}00h)50rSc4@)}kC)fS^)[!.8=*wp<G1+O-0jPblU5Dhhv2(AY=}}}LY}1K%!}0mnI1O@=n}3c<b>>keV@o1^#%nqo.0UVpp"
    "0%w=p1Pe5$]9b0P0@8M>69XNj0Si950%G8G3K4gx0SJh#]-sE:2l$qn1QT$%}%:BM5=7\?#@%n=30Vh7d1{r@Z4!*yD@S&ro><Tf42){t02[v4l@}VG#}s:sV"
    "{8i&g)D1lA1]]eq{3@QC37(6{jO63U>MHqF1DbnE>>K/M^\?JCq0R=]q)<v-63FD-N3jb5V76aNm@Qx*R6Z2Sk}0edP&QNPr@%[Jb2)HjJ1n-M70TfXh5Fasa"
    "6xvjw{-PCG0pZ{80RVWj2(93O}*!}I2fcSA1o[zu6$UdWaVonY1vsQv0Sp<H(gjVa}@Wzu(Ig5R\?$m%1@pYQC6.&jd0qV9+4^-@n20zc-08Mk5jNSKI0$s[Z"
    "3hT]k{$(bO1mc3(1PG54[BDO-0}l<0.VG#o5=-1Q@SJ93@RN(v]860@}V)lb3h>nh[^+D[}3cY)5c{>Z4*8*=F=b=$&@5(\?@qWt$].w7W(<J>4[=P6!<]ktM"
    "@%ED:}to<M83G()3i{OG]ZqUC}UyP(3hKh}015#h}#3<l@T5@n4eo&c1Q1u21]Fuf3j)7E%g.R]j]t{l1odGS0@AA9@R(Y]0%0(Q1P5lf0Utk41n@f22hfmX"
    "}Vc=)}#^5@)!E1&0So\?j@RV<200Ju5@{]*Q@@r02@]0+O3<D>l00AF*@$Lt8@#gId0nmbE0Srf5}#VM{1owO%5cz[b%m#Y}2<ho<@%nP8{W[K=00SAd1o5oc"
    "0r9c40q=V}@@A2&\?#iPR@S9ra%nifa0%eJh@z06BjMcq3]wPGg{X+r)<[wJy1^Y+Q(ivHV@QG4:-nRE57WyFZ6ak6b[/n223H[-8]DkgC]5o]g3Bgcr(iX:0"
    "2mf(04<\?wy7}XR}2k5Aq}6k5l]#VXg[!7W65:{8c7vyh*{AvYY\?2^J+1PoXy1rEyC/\?}hG&}rV.Y-)/#}qRtxFZT[Z1k\?xc/9\?D->(ywR0%wGc4^KtT1@aTl"
    "{yR3Qj[O{^<O)H8}#l)3}U*Y#0%WJC1]!Me2myCg@m\?wL2l0#\?0%FMa@}(&<%nA9e0%v-<0sm-<2lB&8%n9l6}#(@%O}c$uapXwI@%!(m}YlNS0pxT.0FyNP"
    "0%5Vi}vTbt0snYc3J{zd0RN040m\?hN{74oc2l$b4}u3YL0RMM\?0qDV}0@&ud@Sin/{w1Q=@@h#%1p2.A0{Y%O0=S#6j[]oA000312ls(7009i52lxzQ1]FDc"
    "%mVG=1PwDh1)#x<00Jug}VD%22MB&90Sw$)1nrl60qW060SA99@@il2{yNr4{ZNGb0qlo<@TnA4@rrh$1=-V*1o5Ga}#NGf0%wS28Vx*m0r:bm2LOz@}tG{-"
    "3Jgqd0s5Df2N7wn0r0f200if81PFw]1PeA5@q^c4@q(Y{@@SAl%i7E=jMM}p[![zP0{og{2)x}J&{l)O!ClcP1MXB\?%mU)<X-<d-%e3UV{4k=H4a2Q^]A2=P"
    "WgixU6-Fz&3e$<MaUhxP7yMA0ahP3gT{>KW3&P#i}pt-O\?2Xr5a]g)w0%v)jJW1w}[*Le6!CyvjZ5!8:8w9Lg)gYrYg:h2Mip=Cl<)hp5}s3AP@@{1I4/c20"
    "!Dt</&6da>Q6oin7eKk#j[XBg]azol}t*i&@$j@466//62J=o&2kmi:@pZK0{XC9(2JCET<}4p*0@zYZ}2.ic{X$/@00J#D(Kr<}]y<gs@@@Pb]-U2u2(TSe"
    "[DO):0SSr01}pze[bJ}12MBU&=&Lc)}VVx\?{5>A<}p$oj}#(*64gJmp6yC@12Mj2D)e=>K@SiJ30[D3f@T5r96z}\?l0@q#/{5BK[3B&BxjNryA1Pww%}VlV)"
    "@qN600rJx00RDGX}3)iz2iC{!35/F=}V^5}@}M*0{5k3)SU1Rt{ji!G@S@P4}tPKD0Sq=}!CxaS1o5J7@P>3<}XG5f0{X}J0WdvO0T5V3*60Jy}#37i8#G)s"
    "1ok>h2fu/u@Te/d]XCu01owft@{Y*U0(9v}<]25Y0rq{5df!&o@@0Yz1n}D^0%w3d2m=%ojokiV@RcK91CcQD0o:BX@S-G42kuvk0qDx\?@R=V>5bDzg{.SDn"
    "{XsQ-@q=/{1]XMe0000a00i-U2>Fo4(IgX=@R^ug@%n-49rt<k}ty9!@rri10%O8@}#cH53h1xa5ch.g}#uu-2mHFo4f3n72kXr7@}#(]3jD.c@Srl52l:\?{"
    "%n8*71O-i20S@r10S@Sa@@JM75cigC}{Y>Qjph5}3<Eg.d=\?0C4MrOT(<j3-5cQ\?^&}Z<)1]we$+KC-w+&d)$)*^Iz{45jH><]e.@RDf-+le[0[C0v&+\?<AR"
    "@%xqJ1qg[x6CyDv]z!:e0Sq@^qWh}fF:9s!2J$>28#Pyp)HX%*<OXuG}VvVK}TjHQ)fSN]}7\?pr3&7ei2Lvrs>>KGV2<.Z21mC]YWF]!4@nm{$9Q%E8>>rok"
    "joDq+0n4UH<n#4I>jF0t5czXi4/D[68$Av$4d:&f5aQbj&@X#r7#+qV(fe&Y]w7ez[Fz[&>lqg#{Xs:{%n8*b@}=*b+mjs<{3AKR6B+3P{8TXs@R2)P]AUM#"
    "0$1e=[^20){4*0g@Sil1[{}CG3HO94@tL!51Q+Cv1mkEQ9SbQ939$cY0}E3xa{e5P[Ci7R1[.:4]Zq\?/3$%33jQP:w1)cY@[bSBZ[+@W\?{wJ.O4FG$@2lB.%"
    "}V^cb3Iju2]-Ll%@N}LL3ignc0}\?cx{67^2{x7x>@Rc0V@}(-]1Hwd8}r4hy0sYn8}Q+&L}Whr!0q2{!0R=V@761ZA}t{v6]zOT[0%!I%[.KuH].!H\?0qDM="
    "}VcJ{0$ba#0Ter71o/8s0T5z%ZZ6@y}$/BQ1n.D*3g^]Y1sR.nj]UVX<mz*r}tZV$<[nxE(!@9F{W!T}1OI{94GD+l4GL+=00&UT}s$6Z1{2ep1P!!$@q#=y"
    "1nA340qN9l@S@D7]:7x#}tQJ*2M:X/{#@4X].v+T}uEcl0{6:<3JHkk@@-Y31oeI#0Uk$s(W!{h[8j6G}T%N][B#>O4/M{${y)z{0$bzr}#)zW0%N@+0@JzQ"
    "0%[I@1@dv7)]ru4jN@yn[b&d:4gSBZ5CU#f{5$%7@Q*v6@ppr(@R><:[+S7>3.Nkf[^U3]3C^2B}%>c[5+#e3]ATpO2J#-]3HN-(4grQw{xg.L.#fMw0%5e}"
    "@@J64}#mGo7-ieP3Ir]^)G&B)1oX8[)/rgp1pk.K@#qpO2Lm//6+fTl0@R&#{vnsS{4N[>60}ol0o<mE{7^=c{XBy-&{[k*@Oc9vj]klM1}gO35qG/l0Ukk3"
    ")ema\?2n.%!@q$QM0@83*:m]Sl3gnrj7x91*0@r9a@p{-}2(BYk3IUnr3/g^Y@%O/g0S&-a@%5=x}$*K01{B#f0%5Sg0seJ4]y&B=]Aa$02(a-9}VuV@{5+70"
    "1{a=32lK}z@}(*62)fY3@@J920SJn%2ljS3@}MA%}s1EU@1Ptm\?#rDH{5T)[0CLYvjOwTO3h1o277f%y{vegQ(h\?sQ(&Dnf{Y/4z}U*o-}$gf.[C1Qg)!#OD"
    "@RD{0.#o]e}$f^!]8]/Y!B&UrV*/<z0%5Ao0r0D8)/SW@1{LpH1m=l]1pBM9{8r+*@hPPg5tH6w]9rwh5D&l$674^S)^f]J@@05)%2/]Q@@Q*A4fcF2>jFS-"
    "@}(6W}vhc+(f5o@@}==s77fHuFL:-pjH(V++U.%6)*yf:N!3Q^65i=Y(kP-*/w+K7XbMGv9]:x=76T\?{6F6HZ}{k-m{!8}kXAI{e0T!=g!^Ls5{6ZSv3]d>^"
    "7@JuA1T{#B2o%(u<GP79*ZAMs4fjAZoVc6ifAzTD@rhEf@1x%&)[SDK0dB$Z=F3+H/6%o9/#izA[@:dQ/Safn0v3Dm3H!5U4e6e$2&-w(8BII]<8M9fjokDI"
    "!DdYj{:2k5}Y2ev2MaP86CgMs7x:^14e^<s2[444fWYj@5cqh72ko8.1Qttp{4X{]=&iOL63f[I%l6pT0%dA<0Sz&()!2#T]-kjg1]!6c7YUfP)ed4!(JWgt"
    "7.vn{2L)=Z0}(T51mli*@O3&s3K3IJdf^cY8uoBu{u@*}1n.*00iL.Y{4nkdeRiQ6:Td8&I19br@[3(9jNS4n%l.@x1Pxqj\?$MZH(dcjw{#8wL2<Ic(}Vd60"
    "4e6L!)f-$x]z]9[2mywd1}h7q0o:%%(I87J@]Kd\?00-xc1{++o[ELiP1{T#3[cX4O}T$k}(i)q2}TA!61m#f^{x8ln<[/kz1p+.p\?qXzz)df&Z@nO1G{YyDP"
    "@S9bR0qDDI{vXE%1][$]@R#A\?3IBU&2o9vw1Tjc\?FY)@zjP2fL}tHy81O}-U0@@h{}UZV%1[IJ/0R(G]3I.su=H.ep0Rk{]^Ga[23MYfw2kFVf}tHS#}udDc"
    "0t+}n0%wA64fd7x{YZuR75[QD3>8[m@r0b%}2z&w3<cza]8[{(2Hzld{WF%%>L$z+3JyCg0%/hb1Qtd#1O}6Z[Dx3]1PX>#@r-r40[^[Z0R(=>*@xK07B/gS"
    "-SDko3IXxzj]26H00-i42)7F91{KL*]85*{]8w:>@}==}@plrY}1s:X}gnwx@S&q@0@I=*0q)u6@Ru{2{Y.uc4Glq92)G!=2)Zt92NH!e^F%gv0S&c3@}#S["
    "1{K&767#Fe@0Ac1eC/+N}]O*p(I*Y42L[P00SJe(1PXU*4ghy#0S@Dh}2HA&0@}#*}#2QT}S[dM5Ft6w\?WYLPHwIzI3+vq%jOWjn*:a33e8P({}V/1xF=3R%"
    "*pu$[@rymgeZi99j)RvDo6I]k)ha%VeX8L<\?TTdU5:nD+!D76QnBe4-<O:.@7.chW5HN$&G7w{]V<2Y:8$iwd1@#nX9jNSi0rqX65etu9{W48W]v%uEbbVv2"
    ".SRc:g4tTZx=Jfd0TPmNc*6jdahl(60KvBi=+{y=2%vXx<g+zK:TW5[aUIuwjphl21OIf=*z6!%6CpMy2(az[@MjTA2]1cG(iW*<}3(YW3=&F+0@@r51o5Ve"
    "2Nx#82>we@}tEo.[HaOz@RMA})Fc8t008LuY.{+7J0!$Q4fMj&0ToOF6B<<p1mV(WLO99Q3\?qmg2($:-6Cy/b[8a9E5b#dn@r8&]}tx)T}u2<=]9Mie1RHUv"
    "}#3x@3ITn/5.#*}0@%.v@dq%5j]LrR>(i:C2NZa@0ShJ^{vOTO<P2L-@qlM{Tn8FV0SqG*}3)id{YH:4}%:ES}Vv6b1]E@{}Uyp24dTP62lKR#3iZRq2)Qne"
    "1o]584F/Pi}s%QQ@SA&u3h1xb}#cu^3F=r{}r4IQ{Yy=$2mIgv}V=@$}%KE:3Li7R)^Q2J[=/r+%my)o0qMDw{WXEV0seY9@Rcu*0Vyw^2hhoWj]t@{}1r2f"
    "{5+3*)/07Y]91QW0%wU}@ril01lfK(2K\?=#0TeI)}t{@#[+SQ{@QPZU@Tf5y4^A(:}1sHM}VuY)F=2xB1{ttj0R(S$}55S6{6y)40F7yW1Q1(j>ktqY}tyT4"
    "}$*E21n8r\?>*YUp}W-S5}3Er5]znKZ1]OG0}sU3{}#3Q2@@rls[CI$X@}^311Qttm2k!]f@>he(jpJgA(KqOWe-Zd23h1U@Ll8D^+(^$^3EPj+5->RJ\?x8sB"
    "9o41>*-C>P4Fg#i]wYPm00q/W9M*V81+vcR7vo](4ibS<(\?.m{67uTk)>+3<]B{(Z2IP7Zgx37Y%nJ2$*uxT{1NuP\?<[7FX5W8e%}Xx5K:<\?aAjoBdh4/t$v"
    "al@>l6bhO){XKvSbiYiW5!BWv[t>VW]^&Sb[@GfXjq48*}tnbg(f)Zx=i<)V}2yM(3I%I!3h<lg}s$al8$aaq015CS]=anB}@]0Z>)K:n0@*{X)g]Zd00z@j"
    "@rSS50q#Wa@R2%([F-w2>k2FL[dKvI{xyf-%mlA[0sGFIU9H7H^fxQ7@qh6u0Xj&)]x(hQ1q\?df]8wpy1POCQ(Jl>j2k^A/)dyzP{Y{-!0S\?][0q^wT3&5n6"
    "3]4VtjO[^=1NM=ZldtM3[/PyA3j4zo&l6KR/ubon9RYS=)et&dkg4NSa@Q\?v3-$u)&%uNF.S/+FhWFNvkOMO)4Iw/{[BLw$0s/30>KVB:&>a%cSsWu1[H%Af"
    "@L/VG8]\?P<}X2g/!4sP*9TQr@1T.$Ucg%hc4Buf5[-xHyi}K(:{#BlV:RqcJ{A]v-&pv0.3ess(c}8Gv4UjR&>r=8]jOeQD0svu)5=J0>}$eWK}1+W5{5Uo}"
    "1O7HG><FYU(g$zX0Q/N=4*yIQ0O4zj1mD{34eoP70Sixk82B7a)EoVt4f^al0UCzm[AQw.@}#:8}#Di^@S&-E{Zl=%@%[+!{%H8*{[XMo*yTeS5Cca^4I5u*"
    "1oF-21[y<c@@0Vu1qyO23iL>E]xEdE}s1lO)bm*I>]69{:o\?Z<3]qA1jP&t6]7h>Q{wU0\?>k$[&2jNMK0s5Y00Ukef1PXDa4y(0j@@@A9@(txz4Fgef{KJRE"
    "@}+WN{%x6p[b-Iq}V(@%(hgL^4f)gu@Pa7]]Y)s[@o:sZ@q#][3I%=34FYF{@pZlG{X$f}1n*ZH{4!K!0mprq@}(G*}U{o<2M1rc}%:^=@SAi67x$036A4}i"
    "0@&(p@}3Z32M<nN41v2(jpg#A3DAnu[djgR)dPGs}3N!N<nF=w[DE}r3*W:l67MC8]BQSL{v]l{79KjR5bEZG[-oj>@MC&^1lo<Z0]SLH0r:eM0S&(d3G-]8"
    "4LNzt<[!2(6B1*I0po*T4Glbe7Oh-C\?4xO$}lpz9\?Qrqy\?$3:n6akcE[&3pr@s4uc@%e!#}uWAE2KQNr1Tth/{xyJO]$\?d31p$nG5^H0}jML}{1T/Gw]c49T"
    "&LT{X<*Mw90%QC2]5Ql3\?U30#1OQ4nV>\?x<8YH6{aK*t+>(IRA{$$YL/1R)iT>0Ku)DB8[6+gYQ6=e1[9Pvqn&vV8J)F:=%2k]L!2K7BT]8W$z8#6PQ}U5\?8"
    "\?5l}2cL()}>T=01><nuO{wt@R]X/@Z/9}/b{3z}GNeN!J1O.uZ@@9Pf[7[(\?6E-o2\?()pGjOYP{@>l}+].GKn!D1+g{Y*x*>NvTw1&yV*}UQ9-5.V{&0T5i1"
    "3FbEW(/F]M021/23G}T20qMP(}V#Qr01]7%<To:0>NbYr1pj.)%j#96@py3e0}2NO2OlXt@q(T2@QoN{>k+z&1o].u<:^Hn}n.a<]woCS*W$Be}uuYZ2(<(3"
    ")D>b[1pCLC4fk!)7w/x-2(j&f3M6l!0U]TIjOFvb1QMHc3jV*BWbfZD4iL-s4^+8F&T.}o@PUgdJmm[n{6}A5(g<VE2jz)p&qS8&^FSU2TlG^k3fe-e/5DYb"
    "]BHSE<ok8E69M+.\?3<>QU>vG2@q)3hchFnG[DGQN3DyxH9qX<^}TTmZ^C$32*vbjL=I4>V7xg(y>Ojy75D-*j{uh5q@Rb*M2g@Ck[:)pT)E]6e00&D6FLg6<"
    "jOY9P4*@x&2){}m2jI@@@Q*r}0pGNW2K.G!1]efg-qZbH@q#0V4/#>o1{a(b]7S*[3IsMd{wTZF@%o8s2(>Xv@rIJ(1}y$sF+]&T\?uAPK}t*y1@rIP\?1]5So"
    "@Tw=i1ps@j]7@N\?2(s8#{5C*1[acR-3JyIV1Q>\?p3>-Er}tpA}2Ma@c0S8=@{xgu]1RHIi3hsnS0quG^0&xf[j[[{H4HAam1oOC(1[S2&@RV%01p1h!@@hl+"
    "@pQx}&qAJK{4[N*@%wn]@q#->0SJG2%nq->1P!M4@r92(%nzY}0%FD11pUt3]8Yo$0}=D]1PFL{@@.-%2(%]12MazP@p{o*}$/TS}VcZ00s5Yh1pj@f0@@rc"
    "{Xs<)0sn=3]-Lr<}#\?PB0sn&m0pZ0S@Tw&b}2*YV@T5Vz0%J3Dj]C9F1Qbey5Roxz2N*pM>LtR:}VDG)@pyo[}UwUs{WXT&)dHqV01nh(@SA2{0qWib{v[WP"
    "}ud2@0rA8}2l1q-3<LX>3KV.q(hy+Z2MKV7{X>0({5UT900q(@}Wrk^0qMu*}S[v!@@-e}2[^EB1P!hf@%n.\?0S}DU0sePg(Ki</{5-Lv0rqG(@P0[:]aI62"
    "]a\?9l}T$o=38TZrjoU@:&n1+<r8KE/0Tx8a3gXq@>pGBv%nzr&4I4.6@T!X$9K2[QV<^cJ5tG1g1Pw-4[BVR.1P#{u/%h1/W!/>23&gUH1T%6Kr}(.5{5aBc"
    "@QyiMc\?/4.}sjtr[br58}2/W/]-S[J6DuLUa(rnK(KJKKcJ9(M1(5nR[-ynT}UHPO}tY*(1NtWX{ZE=L]tfC@1oX/cFEH(c]qQ%UjPK:!1]E=U%nI]!{4!gX"
    "0rz6U0rio30%e-b1{j]x29n(Z}3c)3}vs26&S6I^[8/h\?]8F)&@TO(7}1>0z0%[-c>j[]L6AXBj1%r%e0t:z*}U5kj}1$0@1onz[\?$Ld[{X$vh2(KVg0%Vvo"
    "0oA}T}UHx%3hB8}1]Fn}0r8uy{vnp&}t7c@]8]76[bJs=0rziX1[Juh2l+2q}IfhvjP+=-]ZQPi0rs2r}tHA]@PagD1oFYd3JH8g1{1-30@z//1Qs[V0r9f5"
    "0%eYp1{b4#}2puM}mV>2)Fv4X0Sq(}0rSc55blh72iCTL}Wixj%ngvT}#M-$@o:)/<nF-Q{xyY]]z5v*1]]Cs0kF:q{4m$R}tZQa}#lJ@3\?<\?r@@0311oOVe"
    "=Jsu4%nh*1@RDP]@s5]m1Pwim0\?c+ej]UoS4fD+D0R=V.@O^+x}$Z^33juC72(1x2@{Hg0/z2yC0r&r5)frT=2(st\?@p*S[7Xvv%1PF=s0}uo&1nJ5@1Qax0"
    "}##(#{5s%&]8OKV@R#PT014&{<]2bS0S@n%@pyo\?@%wraVhKUg3h>aU0rI@%1pbbl1Q1L*]->u[{w1WX2)V4D}TBdU}1U0L@%Y8m}VlDM1nSC>{vkuAjOF<F"
    "2)P@b6:aWg0$ak\?].nT-[GnhJ1O\?uT0q2)\?^ENhd2hP)(&S[-H0P9XP1]Y5h1]!x55^Yuj3g[l0]A=0g3iZLo{xQ/)5b#Xk69!sm}V3M{0Tnw}4eH5d}1::."
    "6.\?Ce]B{&%]-$jg3iyh5@@JTX2($eo1ocTJ{5Lr^}1aTV}uM(+4FfC#0QoWr\?q4&85a/-3TOqF+0J[0@j]klR@%5960TF$}%04QC%nJo7]B.040@RP\?2ls2["
    ".#f=p0d+mXRuubZ0R)090q)90@RMY(1PYhy{W[N!}#W992)6&4@rieW2M%#e}u)h}0@-31}#D*11oF-90rJ614/Vn1@qD^2{wL3/2knb(4/=L61n&l3%m(0Z"
    "0%O-61]!-6@@h(}@#7Ue}$*x*[BDOY]A>D]Sr-4%0kB0UjOX{M0TF]k9rcru@sF(4@RM%80@@G71o4P^0}>7K.#6ud%jNR.XcgN:0q#Y)0r0o3%mlx()*6{r"
    "1QLhc2Od4h3&{Cd@rAA14*A4z&rPnB0@R@}1oF2@4Hh$g@}ul(4D^JX}VWfk<{7[>2la#k4!P&12MK#b5aPk:4*Aak@@&z%0%5x10P./j}$pfZ{0%lE[/mVb"
    "F=bVQ%fL<cjOnA^2]gkn2>yB6@>Vzl^=h4V5+^s*4*\?{$[75-M1R*U9FEpAd^&Y7\?%mCKY}Vv@g!BqIE1N3)x3.2]m[XjYj5*GZIap!hhfdW:H+jktV/Z)IL"
    "9}=)671sVh9Rxi#8UsD&]%+mYUmgZn4cOAz)fac&4\?x9T9}Dh!>n0PR@P>VQ7Y+xy}s>gn\?xIN=[hG!wblYqi73>O+*gt\?\?jMMO3\?WY5y6:>zx7gK&N*1Nb8"
    "@rrSi@oUA$}%9IF5C3Cb9oNwd}XGRu2[D*/0qvDr67C=b{3S^(0r&![]9k](4\?m}6}5fp&{Xb^JcI[8G}3##<0%PIH1OQ4L<{*=93!Ust{1c$+(I*DS3kBe7"
    "]8eQK67tM193hf>=*\?8m}1T)L}t{a2/QI-E0(Coq3P4}.4^0Jv0Tvl#2y+HIjPkAW4fce41mb:$0}D-#0r05%}WFIY@qD^10SiSwXa]!82=5pw}#hcE@S&Pc"
    "@RW3f2l1Vf3KcFs]ACp21owG30r.({0$kC9>k+X)5cq!i0qV=$2)p\?u3K)gz1P!Gk!a%1K{xpP([bJg+4f#RF0r-uo0%eA7@RlY)00JD6%nq<1]Aj)*0#pLi"
    "0S\?#M0[45G@rJo7&%c7{}th*zjLn=^}6yKR{x/Ns1Nua6hU=[w9Q9Q-1>k{v)f%G$b*Dt()fTAf]cK.v)h^5BF^-{@^+j:r&R-9K!BTTC7%}:=4g9gzI+r)j"
    "0s{E(>KH%K6EQ9W2L5332KZ)=mgw>[]B&u4[E2x@>>QCb)Nu^.(CTFT8vW/b(^b[a>QC8#0U2UwWaLA/6B>od72=WXe=2qu3gYr36/UN>4Gkn!j]tfC}wfba"
    "3IB2-<nF(O1oeVl2oH}J1][Pj}W9z{=jfEt[^J+k[EUY{1]n9f=JaQz{3.qN{xfO4@}l&^]YLxa0T5D01}7UoaN]EG>k2tN[b\?[K1{b5l\?UWcs3jD:}F:-Q5"
    "[CpG92j}@>1o!9w2k!-k1l6w14GMnc=JrXG%nhJO]Z9m\?[D/cJ)G}UQ}W8]#}rE}v0Qfyh))&k#jNh[A3\?Ku20r0fj{ws:[3J*ni}vJr9f8VB^(&)HB3J{Fl"
    "(J4p]5DR\?G}T>o&]-T}1]9Vl2[alkX3i{5n1@D!5@Sq(Y@U2kq@Sjku]:gH1(I\?m=0rq].@S-L}}UpN00s5q]6Bs@>}u)b@}$73)3<3Um}VCZX1Q>Uo{vv+V"
    "4f2S3FE}.Z}#uu!3HnMe2Kq4m1[}ye}V3o[(5j>$jNS{L6:(t+(JbYp[Dns:@}WuC@Q{o#0pYaURUoS$4g&<G&%bN-2LO-j@SiCP[CiQX5Z7g@&V!ZF}-rym"
    "%l/aD02kne8wJqQ0TeSb5d^^Y)bWZv69)N%%lPTOckMsV2.rJz@SiMl]CF5f\?wWd01pj=70{6U4/Yh#4}#2^:5V(Sk3zdei2NhNu(iM>.}#ka!}ty0}0TlQ="
    "(K+}TjOe^A4/ktD8W>T9}#b)t(&d.h<mA-7*5H^Q@oAwt2(KY51qQJV4eQhd0$1(09Sl@V3d5Iu)!l.l/ywXt3Jp8201x8k0$>kk4EBxp@S9rD3hBf377I+["
    "P7q*M)!=IP)zgdL<lC$e]7IUV5Hn)[.Sg<%6Y*5p1RoGbWf1[B0QQ6Z>\?bCb]5HbC{w$3X{6\?f1@RMrN2P97=5C6Y8jO!*G2N3!30U1I[]Akv8[=ofO\?NJ%}"
    "1{AY\?01Pkh}$gP}@@&Vn[d+@>@SJr5@q3-*67kh/1nJq*}$o%<5+2J34G$\?#@soec1NDo>76TSB@Q]:%0%e]p4G=E#NG})M[}3GA)Ex@L@j.Rd{3I!A}gF>I"
    "1OSlr(JEhl)gX{N}VcZ0)eM+*)!#$J0q(cO}3N8(}W0-800<b@}{(lGjpiCx8WPAT&%ld!!DWqT)!u+]d[/A#1rNTh}T<OdiVn8U&S}Tc00rq!*yK!A\?#\?xI"
    "cK][37y&e084c-q2\?5qz*5)#=02CaY<]cpf4eF#{:M{mV03ivfa&a9T<q#$)\?g@(Q2>eAJ0sDG]RA0#f.4U/I4bJfj\?$2.C>J&q/>mwO-g:0O)@%wz>}#vo9"
    "&SeVW5%K4F4<6iM2[6uijPLg4@SIx\?0q#@$}tq6j0rq-(1Oz<44gAvz@nE+!3hj2&}2HM\?@nN.D{5t6-1n)-D}u3l]0m*]F{W[Np}3uS@@]:7=7ZZuL}v%zv"
    "]bX5k@IFfG1Q1Vg2LXF#<PTuz@rAOR0%OJ00%4V&0Ruu/{4F:}00APd(<i^[@pfG:@{{D+2k[P70rh@5@}(-$0sfUz}$Zv20U+>v2HMHWjOnpi(KxcT)(J@x"
    "(+S\?>(<:E38X0Dh&}Ay1{VzRU@QPdJG8Wm()*X)H@n)Or>l8wc\?VKSI&iB]D[z\?v[[auCJUr0%Bkk$#W3ad$l&WaWi{sK{16bfv$aOj}i(hP(H{4xrI(vYGi"
    "*-cH&>KFW0\?X+p&)C[ft784\?3{1xz4[b@7c0r:RY0q([)3J7[w{X1L75xs>CcLW372UXS(jo>H1(!2bn[Y\?ae*@nHJ@}2ZEdh.H/{8r\?!}[0gG3iQqp{7V=P"
    "5Bph40}bBA{y=@.9%i5M6y==/&@f&m0tjCQ]a{V$[=PE3%eTJTaljn4@OM]50RMHD}#%hqi1V&u4E<=j}@mI24aU^P3\?r{a1pbg$[cn)H>>Mj[0QPZP5Bf8["
    "1SEs70R)b69PE212<@922(U!o@qco*1FDh[jI6}k1Kh>K{W{PI\?TpHx@)&^T8%[1E4t1c}0PsHz6#/Uf(Ks0p4fMW.@okcO@1g{+]4Bw&/#Ku].u6&k6yL(1"
    "[*>BCmbjZh3<cWY04v9r<>[!B\?a^}+1[5O+84+^v@$<R**XgvY3*5A3@T]kf3#UkxX7}h]&rGnV0T4<.=gzs(50>0i)GI\?W:PqOx}1>l]4G+P25fpQ!&Mubf"
    "jOXWK{wBs)}x2>I6-]lM}VEf32\?PK.2kXL#{Y80s(H+tT@J#.m0UBJ31{>1{@Srre3&pe94Im8w1@)Hr686Y:4GDRm4g8<EF+eT-]z[:Z1nS8%@SK.D1QC2x"
    "}$*6)XbuN7@pn+p%lxW=4^%N{@$=7E4/Dkr4eZed]Atl@{XTTX@sFSs2\?Pa*{XU0&@sY5e@P:d.>l@Bt%mvV%jQ[g.@Q76)}#u&%)/9H>{Y6K:0pQ3W3hjC%"
    "1P!Yf2mpqf}2ZV$%iI2G{5UE00ofHv[^s%]}U{P>@RMJ)@@A62(vF:h0}VJ)@S&z>}Sn4-@$1w$0RDr]0Srb@5Dh[q{vws(}2HD$0r@n#{2DeN{WX1V0rSF%"
    "0%no70%w=70sX#i3h<]g]926/.VZhB0SRS<0}(A/1nzJL1lmSTj]LA#)!#}V@@Jx0{xg)e2jqGT0@RY#@Sie@2GM]B2(%5}@r05%1ml9S}2pG/2(xe[2(iY/"
    "1{+bk}tx*R2mHtg0RMA(0Ubki0Tn@i3KD.l0giDB0@-k{}uVM$@RD^50%4#%3iZk9}2oyA0RD&#}rwK]@s5S9@0J=f[:^7W@qlcZ}#^fc1P[R]1m#fY0Si64"
    "2kwJe1Smi#}Q<O$jokfv3L&G^\?@#9z:smeJ]XPi058WS#}3)i0(!@D-5=S1a2MJi3@$>893>:6V{.AD7bmJ]w9m>L>3L6MO4G(O1{x]15S%5nqL\?VIE-@l2y"
    ")!){(<Qzp)@RWDu].nsCkm)/63*v@%5E5)N{x7l5+nN[<htV!V@S00f0R(lK)goam4G)yw3w3!z]u4KceCYCH<Ksem@UaqQ081P#jotJ+2)Zt3[C@T>1OSt$"
    "2ls@4{8!>=@SiG6}sUa4.#x-s}&8r\?8VCD[@T/}B0SSA8@@Ac13ipqF]Cdi35=I$q0rAo6@%w=d4eZzn69m$j@RVY<}2Z%e3Ku!f%nJ05T>-{R@QQf)00r5%"
    "@@p:^@r&=w}#=<j[yXfC}#D/[1[&2&}#(V(0tk>A0U+Oa3HE(]4ef/81RI4z1Z\?X1jO!*k{5aWC4HiA!1Qk29[Apu40UC86(H>V2]9k9<{1xiyh2mQV\?UErS"
    "1P5-A[=oE06yK@/[7$wQ}1iXZ{5a*P1Rg[C9r=g\?}tQc)1(PsK@[eIl5/qO+bl[LK5D.s[NEf\?s]6$#.>lI$n7crDA(dUg@1[&im0VQ!m<{Q.i<OX]w0%^lN"
    ">LLvm(Iywi0U#T.[^jQ*]7S@>}O4-HjOd+k4EH^()/6S-3&f]7>>>tXdENGI0%G7-4I5Zl5:Y32<{RHB}1B*I1[SlB(xg/8/SCi43)CuK<Kh(^78lf6<}[si"
    "e<f%I0[4=9{Xfmo&KPZg(d(K:@[vT^}vii6U[cu//x@+w1OJJf]/JA:0u{\?u:LdBq)dHfc{VpJ3{$M!&&N#6.2l$IE>lZhM}$o:I4c3)x4GDn3*6SVhjP2cO"
    "(JDFT6:$3s%iHop}1Uf{{6ZM$0}(-[3hT8%\?s*)J(h{RQ]8##h0QY)>}tpT0}tQy0}ty)4(&Ng/{Z33O3<M+n}Vl*G75vL9)!)7!1n\?M=}#l)8@S0c7{x7i#"
    "1Rpqg]AkV%{5T{u2)7wc2k!QU1PwJe2LdlZ3J{Cd}$f^-@SAla(gB]*@RV/<}geLy{wjNW.#]Mq4/%VKjP$:b@T5A2}Vc%8@@h{3{wqM8}2QP]4\?5NC1n8:4"
    "%agjF}VD-(}t*S.1Q1L{1m0-g@SJu9{Z^Pg{w1%X}Ug0[=/+(m3L0mo0@q%a%l!$N0pPZZ2&$%!%nrl50TX/c00Ac5a{E]M@08S+{wLp7}2HK50SR-#0%nGg"
    "}#uo+1o[!#1oXVg2lb2t]-jKV0S@-x@@A5%1[8A^2uj!=jM)jQ4iJK.*xe5U)f-NS(iF1ff7]bB>&G\?}1TjlL80p.*(>]Or)go]>*Vnhz4Ej+s67=^Z@]>6]"
    "%g^vV5.EJH}#Wwo6!e$y@#QaL0v)Zi>eV:b=K]B>b7Qjg=I(Dff7wfu\?T\?yc4\?[m3)Aj+#)lPm%0q0$K^kHWugz-4}a{GH>5Z}7X-o(Wi>k(t2[7E-{@Q/%G"
    "3!m]p:krA>jMbzI^hC5ca(ipn0LZbYj()E+2HIkge-=EeimvgvZVMuGbd)*#J2xw73g-08>MrpU72M1.rr1kt[v/<(2@\?)#dgVp.Z-syh^c7=WM!}a@8u!T)"
    ">mkDHgrPaB4DcMO]*2IU{:2>[eiryK(/C>tkG&SZb<CY!4fvd0E*<#9hReUrdd]e*3L@PA{XD42*NwDM+xq.8ZXx]v799{)jO]0P@1!Vr4e]$I0@r>z0S-]d"
    "<nXY.0p/T^0P@>VQ4wc.}R@j{=jxQx1[8=%{v!)(9p-L4@rS]q5HD]C75OTr69Wy)1SE{.<)S/X{ycD^}t{V%@SSr6)GiQ*]+c^9&^}a9@!p:D.VY@:3hjzE"
    "9Ud@w0@S921oOO>&re#U1q7.4%nJ692l&#>}#uU%*3wws&URvF/.>mK42k0{jM32fY-wJr8YFwe{y(YK]Y)KU&1(Q\?0@{j5)^H2Y=&9820@I1cVL.3+1g*5k"
    "[-ZGrbKtav*w\?($@wT#E@UMKo4flhe[GO2fbMPk51odoH\?2[qY3MfAk}u#o/am!>5apV6[[b@jL2PKxX1RzNB8x5Xn5+DFl1qhD]4InyF0o:))(gD2s}$wy)"
    ">O0\?7@^+jyUorkS<rrJ>}Y$]0jouh](i#\?D1ZR9U)/qP43<VUx@@+a$2mZO40Pq8=eImep2PJp6FFGf%5#+)q{xG*R<{QOY7XUw>0qXkj>J}418wAO@5b(q4"
    "3\?Lhr4GL+()/Aj.9[qbu8#om9&P{QMoaH1J6ajdl]Z:9n00<X-]bWY42l1&0=hmRl(CKb54^A#)@@7N.{WEUy5c8yZ6QXe8]-s*(Z6.*J0L%Zzjp&Wp1Vpww"
    "Y.-=lFD=/ldA*MboF2Sk}AC@kZTifi1o/n<<o=*)0{oK+)h0%Q@vfk$Y#Y9e+}gjj\?Q@Lc@0quh2m6qNXaWqGcn!Z*\?t4@}4f3jY!:/\?y/.BbU918!W<0k0\?"
    "1[z+#5HEUK/fxcL2Nzp)(l=LU)f&Kq5En2\?XDhMI[KJ3]iv0!-}A<]P)FC6Q1LJix5eZh46@qD4jO!Tv3jlnmb*cU<3K#I10@rA91OE]S@{/gM0}t4CS{Q2."
    "1[IZ0}Wh=<00092@Q*Y%3Ijq){5>=<4DmlZ3L\?p<4Hh\?g0U2i0F=2GR@@z@$0rSc5@00&p5CMha%n8PC}27i[}$Hu$)G0mZ2LFC*2jIV>2l1Pd4*R+a4G(Fm"
    "]-Ux<0S8P<0Hz^F@PTBf]A$V$!E}Mw/AhlT2+=E8jOX:G7Z6pt}TKQ{}2IPrfD*Yc03H*OhVT-\?.2>kW]A$K&1OJVz{5C1G3jcW<(\?ynx4*-jT\?@Un9@%P5m"
    "7:PpV1QTk^[:cbs0$$C7@]jp-]92u{[+02d{6o{S6a+6g}3cY*4!*[b1Sva92Sp#D1RQt8[E2Wd@{Zxm{WWF:1>@}QFEhe)5ekfH1puZ*5buRT@]9\?=0qL:C"
    "0znKJjN&QG2M%L}1}p@3}sKT!)dHU@0T]ey@RtpH}2xmE[biB[<<9U}>(ZXT0mHia@qc/[@pH9=0rKeK1qzvz@%c{r}t\?rh0Te!%2lb2n&@]L:@r8&^]:yH5"
    "4*I\?j2JU:&b[7BZ@keKn{YQQ94*I}c2Mj[=0qlAY2($}v3hK-f]7&gQ@T4rW{XK<(@Qy9)1o5h@@sn&iF=k6g)%zXpjPb3P1m^u0]-Lc({w9$S0sGbC2PiQy"
    "F+//D1][.(5C)6T2M:&b@rrDo1Rzpp0@&Ai}Up3V}0!4L3<vax0Vpy#@R^H*2l+Fq2OU\?<%m33M{5+f)5EFW+@VRl#%nJ8$1{>+x3JG>w1{ah{{.S-d0Q*M@"
    "4HR\?h@RVT0)fhzW3hjb[O^kS71O\?)00TnSa2M1hs2MT&80R(%i{:M-ejr0P.2]8I71lw$^V\?JCI2[Dzg8ytsX6cdgS9$^(y[Fzo0/zU9&bkuY>)*n+I5F-)#"
    "\?u13p+Ohw/0#Qp$]aY%p\?Q&>V&P\?J<8v>)p7zdLV!cVYK4*&KE0[!s^)H]Ko9S)h=!yB9^1o^JM00RGI0r6/!&sl>3cLWsi)/&Hz0p!^#4DYeP{VqLz/8V3v"
    ")w\?c7{6faD1%BlR(gm7xj]UoK0sni401nR*3&xU{0Sroa3\?<D4}T9wD4/UL*(\?7k-<mr3Hi$p(X@)7Sx@pQD)2MK+)>kkwY0quJ=1)L4S3Kdjs(J)eg3<3bd"
    "[DX^^@@@n[}tQv35D-jt}UQ9\?01ohm}V3=%}UZre3&{On2O3!70%wVd76j)G3JQtl@Q{cW1o[n(2.zZK0S&nU]8nQ^}#={3W^jke@b+gCjNzOh8u/Kn03Kc0"
    "4^p}+@RvUE^adK]:NQ/8=EhZFd!A#r*xV*k[^D=W2lE6PPBiFU1{bbYNJ*!#[!r8e872[n5fgoD8WFF4}{Ja)20{UN3Jwu.8:kw-@t+2y%lO*$bM*ZS=lJhK"
    "-Ismy0UcQ$5GZlB</6y0{aD!U\?@7+t(b2*{FR9:>}$PE=ZwTIU}}8qd@soRBd&[%qIwW}4jpgs6Y59.B>lqXe8XI=+iT#lK01/u2@tLzh0]Snn5eAKG5BoP9"
    "90c2x]Y+ef{XT*V]AL^b}W0-e3KD7{([urz}R&1E&5G%q4H8Xs^8(yY+jDIv&0wxK@)94!0g:Dc@u@3o8vclR@0HdB2gRJk]Zh2x2<*)k{w1(y%izXt}s:m6"
    "2@}R)1%J4m}%bf)}W8s71RIBu02L\?1je3o8jPCuS3JHXA4ETzA1Q$qj@qvf9@TO!%1OIf.0po%^-Rfrb1[}i@0$o+Q0Odbj}T$r[8$@>h}2H*62j.lX[b9d="
    "2)RpD(jTyad!@Vu)FcwN{5jN^0snI#]ZR>V2{e)T}(oU!}#DA*=&J$J1]!Jh1{sbU@@iid3i]X]Z6jox}uNb]3jc}K0osv+@RM=p/}erHJTtmo\?1\?ia}::xR"
    "jO]iP{YZ)e[AP/4%0^QD}UHG#1]exk1n.]%0@5@8=IdCrf-H[LFDh8v]8w<^0@r961Pnt[69eQF2mZRf00-u62)gkd{YGZ23i7ep5=zO6}VcT1{Yzfn1{s/c"
    "0S8/>2k/5i}Uo<.{67Qp1mMM&2kwUU2k!F-5^EU4}#W341PFC#0S8P{1P^#>@oKBI}SE}Z2Ol}a<]CbR3[J%ejole[0v4A]0s/h54D[P9}UZ9!{4x6]3h:C@"
    "@pP*{WHSJA1$5:i5=8\?f2){>1[=OHT4fcLq].epH@S<28%nAxk2M&xP5+2\?(F:r7q+K>Pe1Q1-l2>d6Y@@ri05d/3P4*rvs1Q2b71mcx@<Pcw4(JNO6@Qx>9"
    ")/&)*2LE(/}tQQ02>!/o3\?sS42)Q802P0vD1(YBv2IoK%@#YGcj[/0J3>8+q9!AUH@}sLk@p{P}}VL:S}1j7S0RuG%(\?7nY0@A68]:5tj0R(Z7}s>o#2M:(h"
    "0S-Sh1Pnt}4/lCm3i.mB>)m83{Ypr]&rf2U}33W0}#(f+}u3V$}VlS^1MGH.}34ib@p*p72m7vI2(s&11PwG03i*Lh3Km1v@@-q$@R3o]1Me!R}VlV.1nh/}"
    "]91<>]8[)/@)9b(jQgDZ0r@5{@rqA/0%w&g1PXt%0q(=>1n&k}}u^9401e(i{5Cc@27V*A@@r8[2k<eN0SSM7}W0x60@I=K}V3D=0qcx(1Rpqc0r-Vi@%!-7"
    "@RDM{1]4Y%0riu9>kLF=1p1=82mptg}u2^R@qcJ}@0S/j}2*V%@qM-#@quQ2@@-q#0@A91@SA6700re@%nqA&0rz/]@@-D70xo4}jPCfN}Y(4o2z8aA0]-$K"
    ")G0s-8$\?>\?4Gug#5:]t)@0r8$3\?Kt:}V>mk2(jSj2)pb65c8kb3>rl!9Q9j#}4p*E}ud9o7wN$p9Rxj776U9G1ow(a1n\?c]@sPUy{vX7[4HJaj{6\?xn[:DLT"
    "59&i&3hCB][-{LG&0^6[3I:@k3JxF=0@ioE0Ru%b%kAXD68&jo)*GH86a+9y3%g48jqW@r{ZbWC}uVM^@}N086ySml%mD0{6+gxW}sa:/2l$p={yv9v@oi[Q"
    "2(aC]5p%VD2kO#n}r^\?x[bq[d@09h<(!Iy89s8lm3HX2&1p:/3[E+<6@U2wt0ShJ!83fWE@0@wY8WMfj2[D}41O8{5(jid:}}76u]8n\?K]7J:{)e)jK7wE>m"
    "&rxw%:[tMl@RMP[1Pl)X1p1X)}<jhmjpu+!)^w<a{w#l8>kUkM{vwgH{xgP]1PwJ63HXh%6c39o@%F+P}>$82}UyK3@[i50)*emH1L%7o(g:]L}#M*1/[5O{"
    "5.N]%0l+9C]t-iQ{$Ufr}@N>Q0p{0>1{1M7]8eF13IBxa2L]kk&M.J9<QRy]0V7w60%n5(1[I=Q@0Sib01Y8u6BLbW:]p#q@RcK90@i6f1{BnQ1GT)DjKSA]"
    "(kXjq{v)y+Z3u3.2*@hcgw.ve*{Mh6=n=T01Jw[m)HglL<urFh[Elic{!6sa0SI@y[9.q9}26d=}W@S5+Rfqaq]Tyw6D(s0]^4@v{rap0!Fml*2lL5V9q5RT"
    "0q)8&cHH^}3kj&M[^$)m^}#f1Wew$.-@l4j3&y8m-()Jj<1u/t9#+F!amYjf[D/Wk2MgON4>+mD})%j\?jN7Yz^hnLR<pyLW5=qqidFz&U0}Wcs9@ce.5Xwq0"
    "4jzzF{YZ=1}s<<4[a54E)!D+O1hb>]5*dLL1{SYS):zyqMG]q2hzb49}$67/>l.E$[z0%3{..G:)dqp*.RBXN}vqTa\?3&@w{KR-k760Cf12R5n(guaR{8j1K"
    "4\?fd1^FTN.865V54GC$%4g:DX4E%S95b{wY2FQJ-<8:SUjMc.w:OCM3]aHfF^E5d[4akje1qyv{7uv.h5^5G)6B&#[bLR>\?]:\?hw3n>#[)G<Bx7Z6#%lMbjw"
    "}Tazp9*W08@vEXj*^Q.Z{@3U@1@dM(}UG<R5Exo<4Ff8!-VBTtaOV@H4bp-[0G=vv].4RN(nN@6c\?V+4(FAJY+PkI3=Ne2=b\?&Oc[/!kA!CB*+bh&8!5EN>5"
    "}$\?-%&XM4%jN<N81nQNn)*5mX]y@m.>(=Lx0o0>)4gJEM4*R}z1&QGv(i^H62lb5d^*uD>}lGh=(Oo0^}#lV]}T:pk%nIM<&%+[=3(nK92[Eai{7d6j0$2Fo"
    "1juDw2(CCi&Q<$\?}uEw})/9d<2kE&:1hB^C}vTes06}d\?0@@q$2LWYu1k<U0}@Xv})/-+15F1^p@pGdB000x52k4YP}ejWKjOOZy2Nx]j7Y1mt0$:LY}UZ*6"
    "(k73c0@q6X]y-y*ZX4s22DOpt]:GEJ0rSk{0@0l70@ql/3KvEx3Jp7%\?3Tn-3Kch40%]55}Vl*54*&pp@S8(N}#^953<cqb1]w0a5a{kj}UY:+]-a+j3&6\?N"
    "5B]R>1]wC}{Z=iK}V3c(0T5z)@ShZ20<}K%@}luT(FIKu(k6r.FDVtT1r>HMjOFNs>mvFb73tP52*.lF)D>qU5:o/6{#@<}}UG%+)cK]y0S*)-9U(l.0RuJo"
    "}r[[32(h{W>k2k{@R=#.}6k%]@S>7N^Hc9+b>}3K!cVi!@s5-a]:}Ak4G2#i3G}x#}U\?=t[b0^Z.3fF}3kKMo73K>j2L[O&isbAE3&PI!@t2q70%4-y3\?<(7"
    "@RD@Q5a*g$)fA4r02+Ob7SVZ@jKtbBWfya7qrzEvblXru0smrjh3I)QL[.va&4Ul3]:gde]6!KR)\?kpD!^{4T.WT}w<{B0s)d.wc1QU.54D[5).$$5B{yMS:"
    "}2x1l]-tPZ86xy[kLKmS0%ERcr*Sv]2]AQ-/7!$of$4nI(mUi)[j:z.4&$69*^pL#2k)#*e=bibM/N(hXdW{lfYzo.aLVFq&T$Vf@w3VWYRH=ej]b6K1Q2qp"
    "1]d=$1Q$te]znT<3HvM]@Ru6.1Pd-(Tnq!#4FQej3KV!h1]F8\?%nJra1Pwb#]:z9d4e*k10@gHI0%xtu5^!YKF=tGN)/iv-}#V@$@rrc43&Zed@QZ0+2>XP7"
    "}T&\?C@Sr6g2Mj@31Qa=a0Si5#3<lb40%5u3@qV][0rhY@1F>X@@@q@^0sx2i].wB.}2pP{0.R0OjI[><4HhE@}ZS3>WCHPCakca&aq2jc&>=!]}Z8aD2s4RM"
    "!Ez=bc)FGV2h}YD:Sx)l[Y{mK6DLE8.S*>G1o[d%^gC#[jriVA7X6Ne.Xj77\?00\?>}]eT-1LSR>59ZEs(>/wl}#e58@2=db[cY:4{#ZMn8U+0T}3uSA[-fz="
    ")TWkoYET/R0pZ{V>(H.K0qwt=3&xPw>\?3gq2S3YTjph]N)JSvH3JGbX9%FFX!gGL%>7h>V&@2>a6:jNc84WR!)cLU!@@J6693l-V*9c$c)edLI%kA4Odjj9i"
    "[d%}48UvbHcfL4N5dNUz5AJ/VdffhW9]W2m{-]$y1{TqE69EXy]y.Xq4e!@OToF}m-QiZu3I<$Gg.Q2V1o[eX(J25%]:}b\?4Ehv%0#03<>&!Y^}pY691IK-q"
    "6Qp9jjN@EA0]KHT}2*&$)^{OS[A{FX3i{Xv0@h)u3K3Uo5A&&.]9UP}].]9]1o]LG1pL!h]7hLy):8Hx[+9a:@@-8$015z(5zlBr1[@X#[^CZh]xboB{6*Nj"
    "{wtKa)/0d(]aq9g]7@s:1@dKw[}NnO>&}Eo]yJv:}VlJ{2MU3-}#W-b2hGz3^*u8a)GSN%%nq-)0@z<62Mr:d0XxV&jPLM<}%A!G}#E320Tez#2l1e%0}#V#"
    "0Q*sB>\?#Bn1n-6a0SR]@0TnX$*4kTO0Q/NA@R^q&{y42@@qlxD0q=:2!yaQ11QUzh1o5Dg@RM{20d+jR}Xw]i2k!J60Tf5k@0Ju32M$5b@R=J*@O^.R@}MA["
    "}0XB=[Dx9Z1p1/f@@9Mg2jzM)0%nF){5aQ\?}2pu@@Sr@w3GISs@UVd7jN-Bj}{av100I/^+kH#s)/9.3]YMwN1OA2/]5*bX5BYD9@@St\?]8P0-]7ipR{X$9X"
    "1[\?@l2Lwt<[DwO9}U{r+/!JBd82j1a0o>r&{6*Y>)=jSU4\?!^O2jSq}>(QRX]UZ.72)7500UupI^T.Ja{2LYI@$tN%0qu9Q1QBRG@q=-N6AW}r{Y}8/1PeM7"
    "0rT8D2hd@q3/{&$2tR+ajJ^*2{7xvF\?X8)>)*wZ<1{+nf}Witl(^#H^}vAL+8sdy#2ljC){xgS&<{.gz[:cz={6R(8(jiZ*0SrL+}1BKd2NgO63i=1H1RytC"
    "@ug}P)!mU03J{Uv2(aDa5Y:W<2Ock1(<rm-68%rY]XPrE@3hBT(iV>A3H^O\?@R^9l{ZlGS}OJ)!6:1*X3<[6X1[AF%1lnwf5jt*H2+oJoZ5>*Gjv!84ke7N@"
    "kiMGekMd>)jiJg+YNIfKjA8ppkjo-bZw<H7ZSSSmj@e0.Zji!9ZlPJBkxyx-kJ$@JkaxvLY[7Lojr2c7Z8y^[k71<7k6>$qkxtxZkm6(9k8XtJZn:oak0.6>"
    "kj>-[ZEnaIj@v$fZoH>CjX8Jdj*8&WYlwxfk4RNDZDIS0ZML^\?kb7$\?kj}ZGZh!QZkDCKmZwOO5j/Ubek4m0[k0\?<zkq1P%ZGb4*ZQC&1Z8SDIZd<nWkFVqi"
    "k9]P\?k7C(AZcZIxkg!!3j%A0&ZQVb4X(!y%j$cWFZUGBaZ7n+YktpWVj#vnJjHVD\?j1I7*j*uT}j%9H{j$h5dY&ts%ZME/vj})lvj@)1@Zmm%0ZQ\?v(Z3+Z="
    "Zg6ptjT>(3k7NI$kJaz0Zh<y*Zpu>KYHseNk3}YjZg*HRZAz>AkChs^Zo%RYk1OXyZc}q7Zt}[]ZjN[1kt1NAZhe8Dk7&B5jLjZ!ko5h.ZD5OEkf9Srkj}LP"
    "kATmeY%DQ@ZqZUVZ8p4Jkwt/Zkn73]j-S3rZcBiHY(EI-j=wG.Z5!DuZL[7AZeOy7ZsczbZg2FFZCK3zkj=2fj\?ImzZoEEFY@A:wj}lPRk3FFCj@I#+YYG2P"
    "kA.rrkmcx}ktG^akLd8)j{[H7Y#XzRZH1jYk8kLAju/d=Y#qBqk1os6jTi):j.1K[kncaTZC734kci^/Zo<\?}Zbf/Tj{h7Lj@z\?ek4t:4Zf/vLZ9q@-ZF$pU"
    "k9d*{j[!b8kvl7Oj=6J7Z7L8%ZVD[#kaTFZZMAd{kxd\?sZob:SZgpuJZ6Ej/ZPHA=joV98!BK*n]8OaQ}%:m&@0+7=]\?Lky2nDq7(g}X2(K7Df}1L{i2KSC$"
    "0nnHH[!7<mgYEx85FU^kdi]b&1oPel1&*AF:]ZwS{0t@36zZIS{Xs<O@0:Im}wdx!]8![lh2uQ^(H%Ap1p1Yf>>L51)JhDw)E*CB0qlJO@})1g3hBb{\?Vr3B"
    "jp98y]C5et]7:cg75P9+2&)ii[A0H2jm9K.spkSc3I:M4>]f0*}4rn#}VL[c]8wty%muGG9q[)U6B0[i7+a[J6IO\?&aQ)w{(<s6/]zF}j[auX<}s1:Z[={oc"
    "&OKI04g8wx3J{qe@q[ho}$fwm(H+R%3F>W-@]i!S9QbAu1%ATH@P1mV1F#h8uKlwe1oXMk}u=JR]7%Ku]-CT3FGCQ70T*<w}%KHY}q@7C3m#t=0)HJ.jo=5W"
    "<]K9O<PLIT66GYi><w3EcBsi{>m4\?69qz[+7um^^@p/HW1[qg63HWs0[:EdL!aSPS0QQAA.RL{}]9t*104[JxaQd@M[*pZw[+aak1PmWo@]:7A@s7N{]a\?zW"
    "ZzEwp<m\?E5@%woa4H9Ue9@cAi]9Lv*@}V-\?{YG(l)!#\?XbM^Vw<{Y]43glgu4H}(e^d=X%2Ok.-1U#\?Ij17jn}1KZ*<PA<00P9&h{5kBl@t^oa0XB2Y1]!@o"
    "(Jc/J\?]3=n]Zh>.}3Ex7&Sx#S90$)L6ZME*0#Z1H0C-b/4*\?Xi{&CV$<Nrop0]jg:)dOEu0RM!I5eTi#[^cok4iMb*Xde/E(HK69)Fl$@1Q$\?l4\?oH0[Dn9["
    "76a{+3f86O2nlnA6yK9E@}+m/2n4vV[^Up23hB>>4DrlSjOe4M%n9n(a{w8T@}#li4*8I6<oj0S5.=}0009-K5bBO-f=]x<>O9)%63QmA5+#{%(<%Bi@TFY3"
    "{3YZw)UzRi@qWA2>yBVR9rbTi4jy]R9Q$iw8%=/N7vR$wd*T(92IPW^}sRw^0u}WM{#-7&0TXlf:OCI\?cL)sg3gFf72}CS=@{{lh>J:C>]9CV+1QjI[0Pa1k"
    "}UgNs[k1xci$Swv74&gr\?nXsu}vA6Q)/R[:@K%uT}7zrQ)/ZS1>Otx@+*d^pe!W*U0d21l>(gFP(<rsu8}+=81@qW\?5/]gz0%X/g)kBwD}TJTW{SJ:l<t=n["
    "]84^fZSrsH^e=Jp6DmU]zWQsG:l}E.<m@iY0TxqiuL$dd@r8%n=oh6$1h+@i2O[BH3[^[H5B]=f(\?H[l}4JF.1n.JJ6A[)-j)R.s0}MGW2Ms-c]-j=$2(Bz>"
    "2Ky:+0rr9k1CcyC1o5RU4HZ${]BH{7%l]{Z1o!Y9<kHmv@0@G6]8v\?Q@]ajv0r&V3[9H.[000c0@%On%@}m0h2Mafl2knr81[{^D}r4XS3&o.[1}y\?r}%%<O"
    "0QGQ.*wHEE1{1]x0RVP+0U$!r@Rurf5CEgx{Xt9[0%5F^@{Ql!(i^d.}SDDMY@VVUZi#G4k0wU[j$0Y5j{1#(ZGJ)8ZYjum"
    ;
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
inline bool bcNet() {
  return (botNet().inputs() == N_FULL || botNet().inputs() == F_LAST) && botNet().outputs() == BC_OUT;
}
inline bool nnAvailable() { return bcNet(); }
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
      pk[1] += pk[3];
      pk[3] = 0;
    }
    if (probs) {
      for (int i = 0; i < BC_ROT; ++i) probs[i] = pr[i];
      for (int i = 0; i < BC_KIND; ++i) probs[BC_ROT + i] = pk[i];
    }
    return bcGene(pickFrom(pr, BC_ROT, rng), pickFrom(pk, BC_KIND, rng));
  }
  void policy(const State& s, Rng* rng, Gene g[4]) const {
    if (bcNet()) {
      for (int k = 0; k < 4; ++k) g[k] = bcMove(s, k, rng);
      return;
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
}
inline const char* NN_VALUE =
    "o&a.x126Y$03zm:03zmx00@mJjokAZ0@@r30SSxa@EK\?y@qV&$@@Af11oXV60rA5#2M+8a@R#{00rz{0}1<<(3Y*s)i%tGX@qM%20SSb\?@Sj0%F:.p\?}2go)"
    "87#u#%nt})]8eT+1]XD6@T5q#@$2I3]p@4ZjO]cV0@&o30rJh%0F7pO%nJ6100001@qV/#0092#@RVY$00030000xe0Tn=i0n*#Fj0C4a}HOd>0SJAf1nq6I"
    "/y4Y3@Sioh3L-7e0S-904Gh**.U@M%2mytg}%T$82&8]oji4Nj2kOS7}3DAM0qMD.)d{+/>>$R=9d^L}1mnW(0L\?X73ITS000rPh@r&5/0(tEHjawmIo=I}1"
    "]B79M0t)GS{YyM)@RuGZeDL}<9%RcN(*<VQ&@Yl10S/S=hu$I$4$}01jOO<Q0@@r30rA93%a7sN00ic2009910@@x5009300rrf20096100JI%015@a1bIR/"
    "je+Znm1DTh)fhP1jO5yX76:Qs8TOjA\?UcT$[alAM&]&U/6V/wg5f7-X3lF{LhFDy#jIRQa2fW2D@R)ca1QU>i2L[w$1oFS65:]L}0@Srj}zJ*ndKj/>@r.iT"
    "%m=/&8gn85jp\?8-5cq}l0%4-Z1PGtF1{B]a@R((%@R(Y%1om#>])+{u)!E4=0TN(+@}3D<01y[8jp7#/{5Uu%0@@rk@SmVv%nJ90@%5M72)gkc0ri903ig5d"
    "%nq#}00R()%m=o#5nGoVjC0wh8U<sk%nR/}@@A5@}Vl=$@@r301p1@c00ic00uiTa1{a#e0sn#h0Sq-)5GweIjeSIY9TRuf4GL[#E{971{=[0)@[)CM{8tEN"
    "5^FdE(DGFU1OzQf}XR*$R0oGs+t$v}i$7=:%nJ8%0@-oc0S8FAF7+}x0%5xbbiX}50rMe\?{w>r<%nr630S@u30qWq]@v/jkjnxf^jN&:V1QK}OdC+NZ[zaxJ"
    "{5%THP/}ry!DpqV3hcKW2mH[F@Owx8(h\?qh}@x@Dj1oW)}O-1w)Fu}V{5U}1r52u@&}roR[:(7)[8sO\?08w4zbk\?oG{5(3984b^c6<#o=jaOyDf=p.#@o\?&>"
    "5>T.l74\?:\?)!Dn+%3)4x6zZ}O}=}Ly\?2E%0{xw!t]9BZ!e8OvajOY3T@@J90@@h]$0F7mJ2lB(90SSi12Mj-70rrl41{j-60%5u3@qV*3@riDd0qKtJjy4<^"
    "[BuOW0rqV\?0SAn@}#cM[@@h#$0%!(c@S9u82z3aq4F*Cl0sFV6%nI^019Fp<i%Uu+}$Zv4}2QY}2lsi98Wvlc>>CnxBkfgB(h#Rf1r)QC@0-Sg@oS5n}2*f["
    "5[k$]jP23R@@A910000b00AzZ0R(=)0ri5@P+3t\?0rEYu4g8$l@@A610rr2#0q)h%}-aJQjokG.1P!P40SSo1@%eL{ch[qO2ls&ei9uI70S*@83JQIj0rJo4"
    "%l*c(01fn91Uu(ejo+(/}2pu{0009j@T58@0quG[%nJ65]8X)>000l5%n8V]@}#*0004MI0rr099mazPjOn^S1p>Ij@@J2$009b#0SJc0@@r300@&l30ric3"
    "@Qbjq<n!]O00Af100Jc30o#Lkjj0%9)9^uz68Sah]YDsajX(1Q4d:$%g9dF83hVpu&V>6u*y:5z@seS93L-pC%aSPQjokJZ0rSr10rri0@SAb#(Jc!Z}#D*2"
    "b9@I%00q8d4*91n@@Jc10RuD]0@huV@fh@[jkZqt7xBgf0rq{e{3Ag^1PXua1{1.%zY!#M3iiNy4Urf\?@Rci(0q+*T@q3oW]HP>wi%kS!{XUA@0ri2$%m=sJ"
    "!e!k\?68IB}cvV{S]81&/{xxEP00Su20T!C%1N#V%9+vN:j01$j5eL3p0%]zl0}W-4[ezMK4/3wt(hf#-]-xTj1NKFr0tj}{0tLFh4!phg6mX#fj3BO<)Jy3y"
    "<m96p}r4A}=(q(+0@qK64jO:S)/H60[}3Jh4/WNw@$L.p4e]kFc)-DHi$}8.!Hp!v0168R}2{EeQW4QG@qlaa67(qX@%n=[1+ex2}#Wrp1qRmP1qRQS1d7:1"
    "j331cgcy<L6-c-c1([w9qWK-D@Qf{{>jo!$2>!Gr)$1*g4/MLJ}U/.9/C%$&>)\?2tjX*iyjUVz)j:)nXY$uU:j=/(.Z1yQyj<P%vj}Xt0jXa3<jZGs4k29wE"
    "YDmHbY+{&Jj[1[sZbuk4jD=e3k3fdIj&)EbY&Tq3kb5zGYUvZ8Z2[ZPj.CLojN\?-TjW<r(j>do[YM/diY>aKajUs.UY@8:ij/5#6Z714djqw2w1T<k^5w##}"
    "F=2iq]bn3{4fcWDFDMl80@8G+0pednjfZiB}VNpx4-WHbFUBcX3]3Go6.ptf[db9Q5/.#42WFZRjacHr{9HWC{c]*::{4+s<QkM^aMkLm)GLQ7jlbrr5gJAl"
    "jF9Bv2)H!Y1{$qP5b#[c[B1)q&nI3J2<g{&{ZE9l2bQCFjf6E9(gK{Y]tH8EF+6&+2)Ofd3<VNe+]u9k4GuwR{9J=*jK9n[0}#X40UkbQ0%/Lq{x4L#\?*{v:"
    "00O>=/av]\?@Ty)GjhRnN9$OqOeGrE&+MnBJ9]T1j1[@hqN&RIi(dt$ocV:eTjolL^{0T4[}we]09R7Lx}$R6R>&\?sq78U#w\?VA@:+.UGOjaN)g7WhR.(GXl="
    "<}e<x4/y[D)d5@&78=e#]B}MA4pu:Dj5ulk5DZBTc:rd.pX12+0%\?Th@U^%58W)hp2>[Pj=FEFajF:c-76$mt}UpEB01*TH0SW=Y@O^Lq}Vc6X0%/+Gd-&Ko"
    "j2Cw>78dtc1}JqpSR^ep]+Y%(@Uunk]6[pxbn[&1{Jb<ljs5SJ]:.b@3<vkY]-i8o}0yGMb7ACs3J+UWk\?N1P]*AxKj52A>]BqN^}W:an3\?%R)868xI86Y[g"
    "0Stm=amo*.]L7u(jGxJO].5k:@R^iP%nBnp5C..1}$GmC4!w#W1Q+$H6YY#:jJOE{]xDZk1T1c:{#rQo5cJpQ@){JZ0SzML{zi(c)mh3ljiWzHljB5y8j.AN"
    "LO2>N<qL>6{%/Sz[aDCH)\?l=J1=<HcjI.SK2(:AS}Vd0O{XMMD1RlgZ1)C<Y@U9AD)F)K-}#:(UjNRE(2)Zy%5bb>/:)YMj[c!yq\?gq{v\?3qNo\?Wo]30p)3u"
    "jN%5>1oe0N}VD#U@Sz/!4\?ctU]NUT80sbL3&R1q>@#uNSj1[6)*-AA-a)Xmg3*()J@r$!a0-fvj]8y30dIAv7]BK/Lj4$h>\?xR*G0Q]JF2o0yk}79/Fl[/Y:"
    "3MvkbeaJoj{$=+Hj4(Y90s[n>3<N]o>M^<G]ajUhbu$KP}%ur+0%OF})/u+6jme.:}xtpPfuS^F@rE]m}4&*B(JCSZi{[Wn(JD>n(P-B/j3MqH:@2X*iS[$r"
    "jp@d&@S&&B1SO#dd!37K*WU75%2eHLjsIKC\?t=!(8u[v#fbkP4]B&CP0sGUl9{k6B0T]y@)qd)5j6.[Q8#{j{{u@A]5+C#8}WL*=85i\?:)I^zI1QU\?W{N&2y"
    "j3a7878)4F2N.MQ>&iAI0rB8m6aCTe(fPSz6AOpF(v>(]jwu=u@PK:g}Vd5W)*Zl34!r0()\?L8p{Wwmx3L:fD{Jf0%jmJ^F4EaDs3!A6oVdo=(5*5a7@}tV."
    "FF!gu}w5Jz*7tIUjt+uk&$\?E{@2(SR@qTFa<{jM/6Aw8p0RM1b68.g:{2>3vjtchO@@SH^2mZ^U1PF&e1{v*h0S<[>@q!z)8VpAW]zq8DYwFa5j=A<RY>BCW"
    "j#ZrHkb/q(kiCwnjRDADjT+-gkaJfXYMe-Pj&m:LZj=\?aZj8zBjR#6Cj{Hm7YKsG/j^kJPYO6dNYsjppk0(Y\?Y&^<sj=s^^Yz@7Kjy=x@j*L]LY#4G)kkSl2"
    "kcuxfY+t#qkcu.>k2q4+j{vJwjB#il:kr/G.S&Ge9*w8m:(q6RYDm[seGVIV7Gw*V6pk@fi@/lX";
namespace mpr {
inline const MLP& valueNet() {
  static MLP m;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    if (NN_VALUE[0]) dequantize(z85Decode(NN_VALUE), m);
  }
  return m;
}
inline bool valueAvailable() { return valueNet().inputs() == N_FULL && valueNet().outputs() == 1; }
inline double valueOf(const State& s, const Track& tr, int team) {
  int a = 2 * team, run = progress(s.p[a], tr) >= progress(s.p[a + 1], tr) ? a : a + 1;
  float f[N_FULL], v;
  fullFeatures(s, tr, run, f);
  valueNet().forward(f, &v);
  return v;
}
}
using namespace mpr;
namespace {
struct Tracker {
  int laps = 3, n = 0, turn = 0;
  int prevNext[4] = {}, passed[4] = {};
  int timeout[2] = {TIMEOUT_TURNS, TIMEOUT_TURNS};
  int myShield[2] = {};
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
}
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
  std::cerr << "mode " << mode << (nnAvailable() ? " (network embedded)" : " (no network)")
            << (valueAvailable() ? " + value" : "") << ", " << budget << " ms per turn\n";
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
#ifdef MPR_PARAMS_BAKED
  S.P.parse(MPR_PARAMS_BAKED);
#endif
  if (const char* e = std::getenv("MPR_PARAMS")) S.P.parse(e);
  if (std::getenv("MPR_SHARED_BOOST")) g_sharedBoost = true;  // local experiments only
  if (valueAvailable()) S.P.valueFn = valueOf;
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
      if (tk.turn > 1) observeLastMove(p, tk.prev.p[k]);
      else clearLastMove(p);
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
