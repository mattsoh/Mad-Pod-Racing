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
inline double roundHalfUp(double v) { return std::floor(v + 0.5); }
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
inline void applyMove(State& s, int i, double rot, int thrust, int kind) {
  int t = resolveThrust(s, i, thrust, kind);
  Pod& p = s.p[i];
  rot = std::max(-MAX_ROT, std::min(MAX_ROT, rot));
  p.angle = normAngle(p.angle + rot);
  p.rad = p.angle * DEG;
  addThrust(p, t);
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
    q.vx = std::trunc(q.vx * FRICTION);
    q.vy = std::trunc(q.vy * FRICTION);
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
constexpr int DEPTH = 6;
constexpr int POP = 6;
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
  double rescueTurn = 15;     // turns allowed for turning round, getting up to speed, and a margin
  double rescueSlack = 10;    // rescue ramps up over the last `rescueSlack` turns of slack and is full at zero slack
  double rescueW = 3.0;       // per unit of the blocker's distance to its own checkpoint, at full rescue
  double rescueAngW = 6.0;    // per degree the blocker is off heading for it
  double rescueSpeedW = 3.0;  // per unit of the blocker's speed toward it
  double rescueOf[2] = {0, 0};  // set each turn from the root: how urgently each team's blocker must go (0..1)
  double snapP = 0.125;       // chance a mutation snaps to the top bots' moves (-18/0/+18 deg, thrust 0/200)
  double boostW = 2500;       // value of each BOOST still unused at the horizon (top bots save theirs; 0 lost 39% vs v1, 2000-4000 won ~80%)
  double blockOff = 1000;     // blocker: waits this far in front of the checkpoint, toward the enemy runner (top bots: 700-1100)
  double blockStick = 1500;   // blocker: keeps defending the runner's next checkpoint unless the runner is this much closer to it
  double shieldUseW = 0;      // flat cost per SHIELD in our plan (top bots shield 1-7 times per 100 pod-turns; we did 6-10)
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
        else if (k == "snapP") snapP = v;
        else if (k == "boostW") boostW = v;
        else if (k == "blockOff") blockOff = v;
        else if (k == "blockStick") blockStick = v;
        else if (k == "shieldUseW") shieldUseW = v;
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
      sc[i] = team == 0 ? score(root, pop[i], other, 0) : score(root, other, pop[i], 1);
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
      double s = team == 0 ? score(root, child, other, 0) : score(root, other, child, 1);
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
    }
    Plan h0 = heuristicPlan(root, tr, 0), h1 = heuristicPlan(root, tr, 1);
    Plan myPrev = haveMine ? shifted(mine) : h0;
    Plan opPrev = haveTheirs ? shifted(theirs) : h1;
    Plan oppSeeds[3] = {opPrev, h1}, mySeeds[3] = {myPrev, h0};
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
    mine = evolve(root, 0, theirs, mySeeds, nSeeds, clk, clk.ms(), budgetMs);
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
constexpr int N_FULL = 44;
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
    "o&a.x11TCB0h6s+0h6q%0aPObjrlrp\?M)7b1]:6a4!!6b1{Chf}#cV*1pj(n0SJb=@1=:q0S@i2@}uQ12y0aejp-=2E*MwW0@0[V@Srua@S900@RlB3.t\?j]"
    "<oSfq6A!sh0@@r3}Vlu\?%iOI6jPCM]0Vp$r0002#000000000000031@dj!e@qDS>0099100000008##0jbQijO{Op-^^}80sPXA1{j/6%m=P{0quV[@sGnf"
    "}2}z(5bu!p}Vl&${51*<1J&p2jo$JAfW{450%fa]@@rca0rr300S-l2F=ada)HPdV{XU0(0rrc2@@I/[0v]F%j+j1JgaSw80rJcb@@8/%1Pwr10ri630r902"
    "0008#}VlDX@qlM]@S9i7@#T!qjobAY{T60w0ri8$0T5D2@@Ac2@@Jl1STOgT{wD$20%wVa00960000940rQkhjONZgFGaZ}@SihX0%n2$5bVCi0095%X#)^/"
    "8Tyt>3jlh9[b@W>1oeu30utIhjob&Z9]N8100960009c2@@A910ri31Kalvi}V#)31owJ60rr91009300nY1WjHA&9*1#(82N3J81kaE\?[-{kW}Vd305EOQz"
    "2lsq}2jWin5Ewpv@@}=[0R!uwjPlDx*!mOv0rW*6]Aj^)1]4]{@S03e3Klw11oFJ8]=&3e:1vw}014oU3}=cgjpqJc}TBQ/0rr331oeb@@@z@%0rrc200-G5"
    "0rJk#0m(Sb9rMlu@@Jc10p5YrjB[mu2Lv(@@@Bkl6cB)*}#)e}%m(JW)F^Z@}VdAo82jQs1m#Y#%n8ca%9C5ojPrO{@2U$t4Gm1h}1B<*[cn)<}VElk=WL{f"
    "&}qi6<{gU[3<u[n0%!]a@f0ryjoj:x=/+ue@@ro400992000000ri61E(5C.3i.7J}Vu/#@@r30009300mWIZjPCJ.\?%7gv0rmSE00io6@@A910rr604\?e:D"
    "@@sF<00062000000rAc10g^Lrjok/-3}*K7@p&^k1pUwh}#lY#@SqV=4Js]M1)KWx<]+>=3J*wd}Vv94@{7@jjPCfE^=grc000FG@Sq@#0S-o40Sz##>!#Jn"
    "]6yEF0TeSa0rrc10rq##@H]mCji65:1N=6<2MA8)cXh=&@0&Jc%m#+(@2tCc1o]ke1pT&5{Y\?3l3Geput-M!hjPk/&{XjQ^0rwX/@S0c3@q=($%nJ910%nJ5"
    "0rr920dOt.3&H28@@A2#0M2TSjL]^ymsUss@RBE#%nhMY{wCl]}UZl*3>-)E}VWui2FPG%)ecUY@%5D6@6)QGjoL&\?a}BXO0S&G40rA8#0ri000rJo21}8jI"
    "2M]Ua@q={1009910000002&6wj+i$kf+T@P00h-@0q={1@@A91%nA3000Ao5@@A90%n8(#1oXV71oFM60b11RjosZQ!FOt!000f30rJo4000000rr8#F3QJ^"
    "3I:#90096100000009920n+ktjo=2$E#<Wj0S@G6@Srl32LFx700if1@}2NV0@Srg00.#P3iH88@@r6104.}:jPCD-@Tn=b008##000000000000000@@r30"
    "0000001BFY[=w:<0ri3003QCnjsqia1fp}N2j8XO>pMh>>)mQ*]y{Ym]6lia2kn#F%hfx>.#f0g}t7E8}uNBgjPLGV}s1v:000310ri300000000000gjGGq"
    "0rJr8@@Jc100000000000d#.Djosj)9U5hI0r7=-008&]0S@o10r9c2j+j9}[G0Dt1p1Y8%nJ8#00i61@&=0xjB[Ht(q!k*@@qdt74}4@3L0am0S&z{2<\?=("
    "@Sh)7@tT/8&T(g[1nh6Z@q>l9jOmE#*5.JT0seI@0Q{G{}#)c31]XV40d>d]{yEct@t2Lq{5tf{1QUhd@<w%IjB$cgarw-f)eyVt/6@Ch{vm+H@nvbI+oO!X"
    "}V2ds3DS+Q5b=La1mmfj%l!y]j23>:06HZt@1Z\?v6BL6Pbl[LR3i7Ri]AUda0$1>$39%:o[CJ1Z0rIiZ3*.7jjISU/:BKK20%6fx1(n+B}TK^<@RVc)5c\?[t"
    "2)pn0}Rz9XAv@ab0P0RQ@\?]%$jPb3\?=!nu@@R(@%@@@&q@@Jf50rAi1F:BJ>3j3$x0qcA}2lj&8@@h-[0}40sjrq/^{V.}.2(-h24boeg5d5Bx0}Mp0{X$0)"
    "@%ew{1Y9$gY5JF]0oIx50GBjQjoK{s68.\?g009M80S-Mc{w>o}0ri313Xea3(JNp30%eVb@S9300rri30kD2Kjr-}^+McS72l/V0yW\?Db@RcTb}5/RsaoRuq"
    "@R(D){sYSf1{+810SF[x(baWKjPuk.8VoWq008$\?@@@r70SJf40SSh#FI#c({WX<-%nS5#0rAf4008##00&(6jndD7=/$Pp@{V#c7Z]HB}3l6+3<3m{[nNZ*"
    "@uily{j3fS(<jvl}te[C0!C\?5jOor[NgGug{wOK[0$T\?]]zP3#}V3Wi1qb>-e+<*M+{0n(0rSAb}tfN.1mZu>job]]82a7h0rixd0SA94@q=(#0ri6100ic2"
    "000300!Efc3Jpbc009610DZyUjLQZd0lCrr0SINH2Kp<N4eZnf}UQE73<3nv1n@h${D6SkFDAK+0quc/%8*WZjpC6gbNaO={XiV#3J/RU}VNPe].OQ.c{[pV"
    "<K&2n}tQK45b#C7[+hwH0%R&5jNSB:!*{[50ry<}0r8]#0SJc30ri91@@S2#0@&r31aMH+!cLmr%m#Y@%0dc+joku.>plJj@@rd]@@Sr70S&r30SSf2y>iC>"
    "2laOY0rA8#0SJc10rh##0g.IXjP2S<)y9e%0SN-E1]v]@@qur(1P]ho/QrA:1p2Js4Glth1Peua0rJVf0C]9Fjp0gp@^+dE0S\?V[%mul[1{a-9%nq%1@q=&%"
    "0ri60@S8Y}@@h(@0rri40kxdFjoV=D!u/kQ@S9F9@15(i}U{D]}#m99mlisFaThHh.#ZkG]-a*)@@@u30!zDQjoL6u)=<YK00AmL0T5J7008##0rAf1v<&Au"
    "0T!8>00930@S901%nI({0aI&KjpfT$Rgnyz%nQ<B2mp[q{5s{(0rri6qi@fV1]QGl6aLVO2M$nf1Pnr20s26(jobDPnbik=00i7(0%!-700i2#0SJf5{:M7O"
    "}#.Jz2lT#a0rAf20ri62@^6amjC0sVh)&C^0roOm0q#(%@@8@#0ri320SSf30ri32@S9970SAc10Sr920rYyuj3rp/6#x$M3&PeQ4!pF3>O9*)(K-BO+V$Jd"
    "64z\?12N/w\?008]:@%n(c3.zXHjP=x57x:Zr0rrJd0rr900SSi20000100031000000d+sn{X$f)009920q#CvjoC@X1m(:21P[O+0TeeX}#)31@%n/6U7bn1"
    "((#wv<RW$95ch+h@rz=)%hmIZjOjl%SuIMB1P8mj}%j1N{w$x\?}#=)9g7tJO{6ufp@oKN[4/MIm@@.A:0+COnjpq2<QwB/&0S=3Z}um=o{51:)0S&n$0sen("
    "1oO(e2.8>i2]A)G@pG*!@yTa(jomed]%TU}}2iax{ZD=/2l1D0@Rcu:cX84J[EVMD(KSE@@qDQ00%ei10zFByjnFJ1)&r>h}s6T@9p}L:-$OHF0sf2n8o@JX"
    "0@8xQ6INWwhUei{3IKe[2B/L=jqdR\?&sV4/}#RDi2mG#9@@A620R(<4@r&J70r-De00968@@Jf00SR)112<dBjMLJxhI7G6}u5F)1M]4p0TX@e0rre=1[\?&$"
    "@S9f87N1IyVk@Yy@oT<[1e@1[jpQfFkpVTd0%:zC@r9Vl@@R&)1nr09<pzK-00<en{Chso<nw(T0%Gei@:phXjPkG(brb/80rAcb0rA35%nS910SJf0}#M/%"
    "0SSf1@^^h=ZYKih%nqV)09HsCjOIefIt3-H0%9#E@qC:+2laA10T]qv3J]>>5c>h<[b-W([CSK>0sm/:{Qp.Ejqp2#h$MI}2LuH]6zYY{[b&vY@{ZZcUlJeY"
    "0%fCF1C3^y6zhE[0%w332<\?npjPCY!5Dq.g008@@000000000000000FEzbK@q=V]@@A9100000009300f3pfjoUV:F.6(4@@z@#00001000300002#@@rf4"
    "@S002@RV/$0ri60@@Si2@$PjjjNyJ39dN}/00Fk:1n8/%]z[Q*0@R@%}Z8py1QURa@U2Cn(h}4\?0%eD4@/X*Zjo>A>SWm]I0S-&61N#Y)00rr8%nq{3:BNjv"
    "7xtBl}3Wr7@Sr90}V=J\?0b+}UjOfQvFNnPX0%duF1[zrZ@SJA5@}#V]{6}]d}#l@&@S-/C{5aZ^@@ASf1BJrUjM@Fwg>C=1@}O5e1QLIq1n@320TF&oeFFzs"
    "@TW==1nRST3jWmG2)6n&@>QRsjokD:>of*9@@Au100rf20S&x40SSi0+}4S{{ZgU+0%FP6@R#@#00i920m>4tjO)a)}4%tt}tsCi1O@op76:^t%nIY^X{<\?B"
    "@@q7B%l]NK0@R&{%m#Y@@(Z3rj+i[se:SGt0SJi6@@r33@R#{000963}#u-@00001@}M^0@@Jf200ii403o7njo>MY]FuEH}#D@j@$bwg1Pe8$1ont@Z-isH"
    "}V=8i2Npea@@So0%nz]%@VxyujPjHU4g-Ew@@yHa1Qk8k]zOW(}s>3=FPx{472Z%:4Hidk00rl9@rhD^0KDjqjM4gy>\?Msu%nJib0SI-\?0r9300002#0rA8#"
    "@@Al20D@@A9TI-C0r&u30o}32jqgKKn*U]a2lPI>@>Pa4@RV*60{o%=U(JnT0q4ML}%s<#].YD[}W8oY103@yjNT%N5pT-T0%nt=%lppc3K3Xl}2HG$]8OH-"
    "2(aLE}0l]D].Xy-00\?P:0ItLDjotM:5p:YZ008]}00ri21o5e#0S&u3@@q#{00if3@R(Y*0rAr60rJi20E=iFj86g30Tw=j3i.U43KqT}{zr&a}${V@)<5[L"
    ">k#Lk{cf8(4&Bvg[E2<d0hi<gj]+JI4by)-008])0SSf2@R={20002#FJ03u1{+Fs}#D{1@S90000rc10gswejPtDw2*AwH@@ix50rh)04/VCc1oee}>70dK"
    "]BqfD}2Zo&@q=]$00ri4@%Yfajpj<PFGL3(@qW6z1(]H&@q)301Pwr55ch+l1{1x538E6h1)ll)@tb29@X+EjjJy4O27V*F@@Dsa2MKt$0T5u4@qlM]3iZka"
    "1][S7}%2Zc1QUng}UfdO0}-uHjqc$KFJ:q91PMK:0{G<>4GMCd@Sh%2(&Wg.}uc)g3Fa!J@r8][}#^]t17T5AjPLD.{ymf6009300ri300000000031m1CQk"
    "0SAf3@@A9100000000000k1sjjMe^k8ee0n}3mW=]9aT$0rq]#%mDu=px}gh4!6}b0SrJj0%wPc4GuFf}PG3]jsRi3h={Q{@raQ31Q2w92(K/6%mbNZ@<\?aO"
    "1Oo+^5C#t2/7F\?m3>0as0p(Fqjo(xn(WO:50%n#N}UG{%1P!M81Pnxa)dZeO0ri64}}%au]9tA}0S0Af}hvC:j5fLI@.i3)}19M@7Sl:.000e#}q.>U0%y1z"
    "]zO<!c/%%#}#cx<]-2i}}r$y+jLN^tvy=gv%nmAR{WFEX[^ko)0r090}4S/l@Sh%510w9j@SAG10%f5o}bO]6jM(jX6tjLu@q}DS1o5G92M:#a0%w=b{u@4R"
    "}#MA(1P[-j]:7S%@rh&]%4e!EjPLDX5b#Rf00094009310096100000FD$-B@S90200i91@@r30@@r300g&jwjQhb*@@Si40@)/P2(Ubf@@A910rA90%nJ93"
    "0rSl4@}MT7@@r9100q#$0*HsUjPb-*{6}c60SNYP0SI([2MK(90rr931n&5@0098%0UH.g/8M3P0rii504LlrjpG:]-EO3&2l@oK4eY5\?5+Deg1O@i3>&RQp"
    "}27J:0{Gs!76ayq0@SMf1z/D<joC]I}@!1X0rr2<0T5S51POJ50S-l24$trw6z]Ah3j3Cc0ri610S&8@03L1IjPBNt<NAiH@@A:$0TwSa@@i32@@A60E)+N\?"
    "}s:ay@@9030rAf2008@$0Ag06j6c\?JF&rDT0S^^22m/C<].5H@{YQJ*8w&5A1on]N{%4NJ0r&8$3Jy\?v}LBksjou039uCjX001ey0rrc10ri3000963FRAgD"
    "%m#A^00i60@@r600ri300qoH^jaO*)sKP512[z}]6.6N53K#\?m0poyUe=L}R%nrh)a+)>Ba>uC&00SGd1<2E7j[]3-0qMS]@@W&g}3^r22MK&60r&=c7Z*fm"
    "@quJ}{xZM}}#V(%0@-l6@:w3ijotY+]$HOd0%6891{B(b{wU6(@qlP$08BFZ5G}Fd[DF**{Yyx}0q(A<05iQ.jP$]&2PS<A008S{0rAf100991000000SSl3"
    "000310NMz<\?#0fG00Jr40k\?nfjr)x7F.\?Ez0r\?bI=[+LD@rAi31M/O1g59an]7-K%2PTGO2LOJ3@080I[&7hbjok&=l](Xo82#zI[QSav[EL3*@UkLutmRhp"
    "0%wJ12r<jb]:.At2l2Fy{YJM0jPh\?wmOTc\?3&ZI2@P+v0[=/c$@rSL#-lA>0c\?}3*{.0(84*Aal}t*u)0QhxujPK^/0W[6I008@&0Sh&@@@r000rif31+u:n"
    "4Gl8c0%nJ7@qV/$00q({0mCTyjE5q}Z<#Z*}V(KD1PdV#%nz/#@@i5@@TwV00SSb[3/qjR{x5$J0qCHK@8xm/jp9H<YG(V40@&qT@Sixa}Up0[}$pm1>{ONj"
    "7#b3jF=26s@Sr2$0qDf!})oP>joBTq2JbNT@qS>w@#p\?03h%@50rq@#7WVZl3/O&w.7C:g{YQT00r}&)0^4w}jB)$yeAX65008M$0Sh@$@@Jc1%nA30@@-i2"
    "00i5$@@z&}@qV/%@@Ji30PIsTjoDb=1S!fM@@N=n3j3F80}2:<0sYng!EAkS4G!YS75dR7}3m371]nr133wBmjoLM/4E:P4@q)/e2ls(52m7Fh0qMA<^+]Yj"
    "\?txnW>[Bj01owG20rzr+0s<qFjo>PV]z!<(@@IYY@rr930rr610rrb#0@@x20rAo21+vD*1P[(d0rSr4@TG+]ju<kU8VQAF4*c$Eo8wkb%nh/.@$bktjno6-"
    "2(%L@a42#i@Qx)T0*Ot*}c1BBj+j4u7Zg0t@@I{a0%el200001%nSc000i91009001m=yb@{Z3&0Sz#$0p0iHjPB}DhdDZ)0r0653hBJ7[+Jv(00io1FYbWh"
    "}Sndr(g$UY}#D)50r8J<0}2^Bjg][[i[r)4>O4SX5=r{C[D/l%4Gvsn+j$JA@S&Sc=f$#qbk@2E0%/tn39W][jp\?D8R6Vlz0rB>-1]Ox40S-r3009b#@@8{0"
    "@R=/%1m}7N!Dk<L00Al3@$-zSjPtG!@S0340rh!B@@Src@@Af4%nJ5$dr3vJ@SomA%nq(%1oex3%nq&@@&o6Mjo@a8t7lXQ0%gjz}#(#$0seS7@RuG)rv!a^"
    "3h/KF[=F:@%nJ31@q=cY0[m@HjPmaD&l.-k007cZ01w919S(cv2kmY/FV{u-bQS*^(K-^\?4fDL61{-]:%fRgLjok:37+tW/@@SdI@@-o3009910002#YP\?^q"
    "}1jvI}2*S$0rr910rrc20na5AjPU==4F{b9008]#009610rr91000000rr91000000F8#}1]!P6009300d(Gti$U%Dq^e8P3KG*.>MFiQ1nzYX0seYf1%Sv8"
    "[^vwx6N6V-{r:9M}pLiB20>@zjojEu\?S>jx0SRJ.0q#(@@q=(#%nr61msMY@4/Mv{0r-x60rr920rJf10laa3j+iRvg863L1PXPn2kwf11onA500063@R(Y["
    "@@r5#0quf>1oXV60r8Y]0h%rajpCY9nho}[1[-=a2e.u01{Ctp0%/hnF.^@#}3wT=@pfmz0@z([@0&x40=1XtjN<fX+WI7W4eI5m0]@U!1}7>t01n!-7V=b3"
    "2MaY49gchM@f)AO1a!5+2Y6kJjB]0\?}%9nR1onVj!eYCA@3RNJ@%wq&*4)f04GM5eczdq83<czt4Fp\?H4zO=kjoWC-9F8W20Tw8@2HSzF@@q&[2kOD9@1Y}t"
    "}3c(>]a0hM2)gCf1P!=a@xXpjjozn902k.y}txlJ{xH46}2g*a}WJ/2OuyGq\?(CHY}%<HR[brd=}VV-)2*1.Bjrk!Z>7g$c}toI5@}VP}0RMD(}%%{!}s9[S"
    "1nSDc5C+F*}#DM%0T4V+}5cVqjoLV!3h%@9}V8fT+)+fT%nJ9300Jt#]-L0(@@A8@{X>c)%nJ620R)311<>DMjB)!t3Jyec0rz^60S&e#00ic300030@S8(%"
    "000010CqV^1]ne#@@Ji3%2JOUjGxJj%9>jW0SRWI3\?S&Y@S8&@1PFD12L!w{0rSo30xPo)h47#q@@.#}@>nX}jFj\?*cvVvc%nqB21lY^.}2HK00%5u63&{ni"
    "1QL2h01eYq0S@r10@-b$1l4k<jB[vGkj!{Q0rA0d0@h&{@@r620ri8#@@Af2000300%[(o00i9100ri403Xo@jJVHV*6mVZ8WH@<i\?jW@2lB#30@q(MS{]JZ"
    "@SjUJklI5!%nJn<1OQ<{dZ:zZjoJz0<ST*d0q$EB0@\?S&4f3Rm@@.@$34SF)2{F{L5cz}b1{j-7@@}=<1e0>djN+xPRR-S@4fq1W9r3YN00JPb@r9c4@S<zP"
    "@qtWD@r&]x{5Ui{%nri62SZihjqgJ&4V#m!5cbf3@<-KL2kXG51q7zt1N=u)3i6Vj>Ls/F2(BO<1o5-g1!&BujP$-J@nW\?S000930rAc00SSl30rr910SA90"
    "00000%la}oaohGz00Jo3031Csi#wSEVM}<(6.UEG/)$Bn4d9#=00A6v9PN7:]A3GZ0%WB47x%c&0pi8.1nxEMjoLJW)c>5Q0rr930SSi70%5D60002#@R=&$"
    "00931%9]ka3JQqe0rr9102QXujO{zT\?ZFE51P/7/1qgZ}4f=Ud1{K>@ii<&@(kt.l{BU!l@rAre%nIu.{lI6VjPCG+3>rpq008@@009920rr6000030N1t8d"
    "@@h/<@R#%10rr91000000drS*jphy=F7s]01QbnF@)P%g9@=9p@RDG!3\?0=&1PYnI1>Ri25b(e9}#(S>0PgIBjoL(:rhxY#1o5x0@}=P)@qM*0@q(/[!iATw"
    "1Pj17{5UB00S-r3@@Ji40ka34jotJ/4b}i^008r000rr8000320ri5$aj{$&2ktMQ0%eD4@@Ac100rf20LEU%jrsQ+6Ol5M0SjbL]:p-[<{Ia:@%5Se10dEL"
    "@qWr76WY1]h$dsW@SAu9}%[LajOESsFFe:.0S@CS{2>-n@oa7=@$L+56e)[@b%l5f{7l0J@S-u6@%YFq}DmB@jn<VVxAuj34fJ{[@kE7.1Q1S80T5r23buVc"
    "]xAyW(mzGs0qV#]5/InT}7wxqjot9V\?VB5!004MR1m3f/0r963@Sh@]69G0Y].Gc<0t1(63JQkc}VDu*0PV(CjPjHQ0V.mv004PH%nz%12M$bb0q(]%a(i*<"
    "%m(xS2lsz#0S&r6@S9930)sR7jqE]07UOJ-}VE^/(llU{3hT@11<!C94/V\?70@&l9{4lV48V6:s0qEGq%6$02jPk]\?]8]0<@qMzv@T/Is1o[/80Sh/)[s*3M"
    "0TGzP{YyP%0%5G6@}(=}@V1&7Zqx/CY##9(YTDaKZw<DAjq3W3kbrLsY:vYKZv%&ikfigxZ79dzZh7&7ZmCp9Zbij4Zc{3UkewoDZh9]:ZXjq*Y[>#kZ8u%e"
    "Zeq/>j+5oEY}Ze)Zr/^&kk+4FZa:(SY5lf3ZrFhgk0%g}jepBNj#@{\?Zr..gY-a9aZO%lDZ9[K{jgHp>ZqXwTZb1c(Zrs(rj&:vZjV3z$ZhW++Z1&S1ZCBLu"
    "Zq)5\?ZWkUVjTO3xZ7vajZf[QcZp=@NZgG<&Y<TrQk7f&FkcL.kZ5RWBZlwMBjR8vrYD*pHZ0elxY!DznZnp1rk7*JzZd2SFj<uP>j.Xw@jn=}CZ6m1Cj$spE"
    "Y)ff^Zn\?(4ZawcsY!yLRZn))DY:wyHZi.eLj{c[Vj)!3DY{)osZ9)4jZgeTxZu:[TYFbv1ZHu!sZsv$wj[Y9]ZDZBHZlf.qZodzGkja-5ZgfG)Ziw@VZTZZ:"
    "ZLf-eZ2O(-YUF>^jY-&AjVBQ(Zn@vPj(<MQYUeIPY*wXlZT(H}jR-qCZeWd:kc[bVY+2#kZ7LK9YqwKhZM8}VZXs87Z9[l)Z690kZv>xZZl9}zZjI^+ZIN/K"
    "j4L{fkeYDnZRk7VYS5M-ZpkzWZjX*<ZiBTRY#BddY=1.FZv[Umj&dFJY\?r:lZvxd@YN3twj:ZJqZKA&8Zk{:MZ5ZA}Zp=F6jYGZ/Zqr}rZ0V[zZeWYkZxd<6"
    "Z69d!khG3{Y(Y8sZP..#ZSE6sZOr*yY#1wPZE<eqY>yj5Y/S#>j-d^DZp+&YY*q5Vj)&(QZe3IOZex9MZpv<-Y\?]#mY-M.1ZoRBzjR6FOjMuqm1Qj$C}s:pU"
    "(<}%:=)[z#1$[Hn(&EN%:TE+<[%*6)Z7]&q9$eqm4F\?7Q4/qg15Ca#$&i13>*4jqg0sm9@>lF&=\?u0rB0#7dS0q=Q8S$b^j5D13I7W7R$0Td3OFb/3F1)>^W"
    "}XORP}2neK}@W}.<okeS3JoSc}z/w<6zST@Zvg$67z#MT4FymW02kXe(g$2a@k)*Zf-6JUj]kzl0Sr5{1{a-7{xHK1{5Uo#0rI&)[=5Z/}T%:({w$i$0S&x3"
    "009635=QU<1PO=71P!S9{YHE21Qa=a@RVo.0S&Vf1o5w(4Hh[k0@A07}T::/T]wU8}2*-$2Ms-1}1%)*0Sr37{6o*X@q(-#@}(=()GrI11m>NU0s5q%0TF-c"
    "}tZA}1PwG3}Uyx*0$%n/)fry<%nJAa@Ow%NjoL&j[:D8B0ljZa}W0h)]a7T6}1>f^4GuLjF+x9L}#V^c0sF/h2kef2(hH8N}q%d3<o(d<)!^aI@q(*8]9tf+"
    "@STeq@SqSN}weYg[+-se4f=Oc]-$G]1pLkg2LXVbQXa@}2kOJf@qc)95deKt3IJ#%3iQt3@QGZ{2Gcn]1O}({}Vc<03J*IC1O-l63jNBl[cey(}%0Iy4%b%P"
    "jN.TP(mR#u)!2(B}u)D5@0&b}{6YBPFFep<0swVq@T4*3[8TSp9TI]G)/&mT}%x^d2l2be(hpI#}W9Aa{WFmX{y()52)yT}10dan\?Vi(I^foyRf=7J5\?2^]["
    "@}3oMbM)311Qs]v[cl]a1muS]@RVJKZ6>hF[clrd=MJ5e0}=K14cNx+3^[[V0Q6{{3[0ka\?q^Yf=J8O[3s^\?vjO!Y!2M:FT7t5l\?0S&I+}W@@f3I0DY+lFgH"
    "V!(q^[cn\?i[=wmQ>KwrDa]W3h3f-9i7Y<Qs0$tC6}tp6@]+mS01mDZi7XX0X{-/Fs7-J-4]-+y4PAU5P75!y669!jg8WL)i4fl>a%mlB86:Kdj6.oSV7Y:mH"
    "0uy[K1oc*S(*+kI1Q>.i{Z4U%0seJk3*bz4}@4Fh6ZjlL4s@2Hjpq8b@1FF^*4%g[0UCd}2{Yiz@11tY]9Cx%[dD3l0snVG{6gji0ULng[Cq8+{wLSF=G(CO"
    "(d=*Q)\?Dcx>)VUP}wf$S1K3us.VY/52LwD2@TPbi4+Rc&{%63E}TCjx6+Y<q777AX@k5r+}sT*.]yzU!@0@#4(gzU1[A*5J}TaZS@T5=o2)pmY0%n]41pZm-"
    "&}R<F0QQ3-dfFnAjPbuZ[=xs70SI(@\?X2LY%nq)3=>crl1o+a#0Txba]8Zcp2)ZLp%m!7&4gh}k1o5ra{Y.2)0SR&*]a{Y}5^nHs0SS6a@Sq]Q}us[s1)(f="
    "1Oq-]@SrMp@REcd1Q1#f@@SAe{X<gR}tpo!2>[61}X]tG0r-ib0r-vM1PnY2}#cY$@S@zW1S[uU@q4SmJ%tQ:@@qx&0}VD@0/kXNjQzb*@qMV@3jd7v00ii1"
    "@r8{10S&u9{5a{)1p1(5@SA39{5k0%2N7bc0q+gB1P5P90rii3}3)x6{X+0%}#lA%0r93a2m7ne1{jS80S-D80q)f4F=b3u@@S92@R)c80TnM9}#M*2@RuS*"
    "%m#($@R=Z03J*Lf2LOl7}ul&%2ls&f%nJ6200iL@}tHf\?1o4S.0SSc6}%K*\?{hZNAjNIFLe+#<C10VOf{c^Ew5RoxX2f^]m@PK-L1L<Z:/5X1M@R9\?(]b3]Y"
    "d*Kr5@qa}D/r1Gn[Ba(=@q-X$>QcxB]WzL5h2)-}6Ae{F]:hid2KQQ22l.9(8q$V04I/=J*xc:W2K}E6\?#Kz&\?qfsx59RZ=(c/1C{6n$<2lHvO0{e})d/Y+u"
    "7Bek.b>}up2Lnw/02Lg=8v^nO<=S2/joMn)\?usz(anl#S3)t#P@#QCq@%[{S5:oF:4\?fVN2jri5&qSf60qD/N68Rn34GlFe[2(S<{yc<2\?2lpy@YZ#7(>6oE"
    "0q#A${67Td>go!b1N3do5Aro=/Xt*4[cOdK2\?wR+{Xb9T*X7B5<Jui<{S&m74/#R53Gq7f)G<Jm^dFaA)!mw01W-tq/o#3^5BZsj5^WX1]8]J+}h:cqjNSvT"
    ">M0mz{x&RX4FYYb1oXMa1M6mR1lP%m3KvWw4ts-O\?#0/Y{y3{0@Q/K-*Wbss{..l=@%wt[=d}Qq[G5kLR#cig:]\?b={WPmg72GI<2>eJa5=J1b{wBKT)hh=7"
    "]TLXr&0ZH@@}Mc&}s/BR0iLe}]BgNk0r-SZ\?%H<O)JeLfZV)-A@ZE7[)bmD&]y\?XA3ITqF}udR@1qY{VjN%9b2N{h0hWoM>{6ZJ/>j!MP]:ojuFEz+V2[=>b"
    "]8e1/0o9$B5Cc599@c0o688dQ84M=Q{B=EE6CyiH]85az@1]tt69OW97x%Nt3Jo#y<[nDWW\?oqt}T+da0@.]*bKA8g3i6(j[+wyn00J093hjz.(i)!k(HBrH"
    "(jSK\?]7&4w@@8&&1oewY%l6:T4Ku/f&@5-J1oMpi\?)LaIjr9.^{Vrp<].wBX@qlM0-$d!>2OLk]}@!q2@ziW<1@)WtaQuB6!:8k/2>X/M0$3uF>O19l)hNd)"
    "@{GvE!eN(<<gk^m{7dYS:P@aE].n{W)FchV{waW(}OAv7@rAW84h5@K]aIw@77H9H6Y/xj4gr9P=jy136:9.O5A@6F!9d7\?2MBF!@(>c34*&164H8R6FJTwo"
    "1ox.z3Qh&\?jHR6(4ndVAXb(Tv&uC+}d<<vxc4MH20knao4J3o29TgD%)GzK$)\?$=+j%pMB@]7Qw9vQI8}[g$D1}}^U68AgJ^82$x[CH/G@#Z8D+Nr.K%jczZ"
    "]-t:g1rYw=[a%M/1q}0U]AV/.9)9Z414Oaxd:wz7)8wwY2h5hL[^4ex*s%4B!A1W:f!AXKi[$[M3&*}.[G)i81p16a0@MHLj]+G.3I:q]@rh)40$bqn0@q*6"
    "@RuD#0sjnG0riG41Pxqs0TFG2(ho#w3Lrmm}t*N1@pxK%%nhrU2J=9&1{Uhg1nqP&0q#&V%n8##3gm/)1O-b&@%FVb%lYmS1OR&}}$Zu]{XC9@0%X&g1PFAb"
    "@rAcu@0iA9{Ypuu}#k%Z{xpA]}$]<i1P!V90R$^V{Y7u:0T5b%0%wF@1]t9-jL{wK>K[&b[A//M]6uww0r]$D}VB:>)fr{>{yXkq0%n2]%nz#)}(oF<!C$<h"
    "1MYWx2kX#r(Gm*){xfEV%lxmQ0%Fi434S6#}##Y(/8bF1:R&hj.SZ<$}t{)i%lHAU6zgRO73bYg}2yuZ]9jpY4!6As].XW-6DuxF@Weu>>LUL\?}t6**5bk#o"
    "2Ompo5cqZX\?xRZ({yNf057FN(jpqEI@R(it2^:A&)/B)M{tM8=@pxjF@@@Se@0K8N4Gltf3Kcw9}t*M@2MB622Lxke4/(!i3G\?r[@@@]z}4&sF0%OV5@{7vc"
    "[^brP3\?jp#01nC[}uE9h@rrt<3\?s(n<OdVR%lG<-(g#$04!nVY)Ex-A71@-}1mlNh{U=bJ2>dr&(&N4:@09PP)Cnio3j%]L}2g9K&nBLo7>qmAjqdC(1njqK"
    "{ZNIL}YUzc%nJ0k4*@p9[=v$r2NRTB4!xO!2>no1\?tvn:7.lrr[cQ0j4FPtB)I=@d5aH4^cl0H175[WD[+}NV2j}fC(DYql3JP>Q3<(77}tnes[C8wJ@]A}&"
    ")^E*.4gh.q>[BBd{1mLw]y8=03hjBJ@@I]%6y=Fm]./Ab2P>]40QVru@RV]\?]Bn[w2Ol^A]dd!]jotDF1od#D}U6)-0qDrF)e&K+@So52&R%b/)/iR8@SR<f"
    "@N.eR2k!q%4f2!&2ic9:}VvDm2LXM6@%nYj[!{SZ}t*A)0!yE!1QU250S0b$0qD9(<nF/T}#M/{1OR#R&R.g73&Hej{Vrg}@Q/QV].n7Q00Jh*8#YTy]A>c]"
    "(IpCY0@h&}}#vcB0%F.]4ey4#^/PHA@n^+Q0+orcjNS*O55dx65x:82<}lbv{BkRj{@-4A0q=B1@$U\?b94yQO{x/^<@s68s@TGCA1)$QK=hvzF1P/tk@@*WL"
    "]7&aM@nEIQ^d(&L68r4x5!b6w}s@\?T@SIx<4ex]p0s^)25EeKm)!).I<pg./77ZG>%jMDC%h<(*8SG#^}2P)^(GX9m>NEsT75EaB6BtH6}$pS^%a7TV>K!X}"
    "cZh}rjO/JB*8O-P0[(65kFvE6)c$G=+@U:81@VU10}Y{H5!Lup1{+FF2Ph)Peb5(tb>&eW63yaY<o%=!2g%7]&{d0xa)Y12\?3+nI1p24{03:JN]CFU=2(s&5"
    "gsXh<FZ:$#@%[LS*YcZ8}R-b7>&})E^K9Vw4iMY3&@fj^4fNpm&i-jy&n<\?p/EPMc8rpXJX=JGD}WR#*1k.n%8o6D}jOoM/5c@7N@&thV4Hph*[.J^J3jmr+"
    "a(rz6}vi9w>>B=/@Qpe+9rVGQ{ZEw>78WeA!!wPl}0e}y01F5{}r(QX[+0H:[=n!.\?o#b\?(Hs${4E@(:>jX9b%lQS-1MZ1g]9%vP4&bu$[I@n(3l5dl<%x4Q"
    "&#Oi%*#[g(}s+q]2oJ:47w2lH0S0c{aO2%D}1kx&&2R}z&qrcO8fLLYjO5:N@@@o4]-jN!2(Ler!cDY[)!V>K8T^\?c}tZN32mQCm00RYS]:7Y@[+-yv0%[k."
    "@oaH=}#WPF[8sVJ@]i}M{r<0k3jN)X0@JtW@Q*r$1[I/{}0]75@[X)%F-u512O!GSV&y7c0r9r32kO8X}t7x\?@%GRs0@Sc74!GY13j([8}j4G:0sw2=]8F!1"
    "2)x]6@Q6^<@]hue]K}A+jN&s4}skpy2j8u^]=zJ[@lk&W[b<GH1ITTu]Cml79Th9d]8w1v%kaaOhV%/:2N7>6+<>DqgV!-7)C!0V[yE1g)HPja{:CO}<PUqB"
    "a{c*12kuND<]jxAFDLVQ<QzE!%lYvp]Akr&:>W*Q@$(:42LMpAbld.-[E+W8-@mEA1OIQ9)GiR13JH!E1kBmn4^%(P4*0ag]-bpl7!W/&jo::o@}lu^3zEE#"
    "@T]Ip]8Y9#01nn\?1]XS7{3q\?]0s!=84gJjc@rSG7)F^B\?2l:/31Qj#j{3@j!@%5G9{w+CH]z]c#2(1P7@TY2n1owc1&RSJ:0nX1N[CSK-0ri5]3Kb[O1Qa]5"
    "@0-q>76jgs1O@06}2QNi2oy!*}40b({YQH60SA0001PeC}27o+1]wSj}sBE/}tg0\?6ucShjPb3N\?#<WV{WNbv@SJC:F:S:P}utTH7xaWO2LF#95En4t2OvQL"
    "}{kfo0@&i01q8pt7x1Nu(>6Wa{4mex]x=$g}UZs1[B)830#.49(I\?1W@r-/l}s>G{@SrMd2iD4b1P)Gp4GuLB3LgO+%mVi!*WKCd{XB.i)*!d*[BVIu0T5/n"
    "0qDM{2<IlG6aCid5gkfqJT=r:5b=t2](>CUjpZ&.@Txkj{6pflao89q0qv09{6hc11Q%(g3\?:I+5eLie9RXB$@TDjd8uZlK2NZIf>{u8F1lP<K0RtNZ5D-7l"
    "%ks7<>)u>B}s9!z3hrVR1SW4n1mC*Z3IA#>4/=qa@T68p}u>sK{X%{+5C+(i0qLTE]=B8d0S@BN2(JP{5deda1pa+Q@SIY-4GCe(>(yOT1{LCf2(se(3y=6N"
    "jvx.>4Cm=!0r8V]3G.A[0RcBj5Z]a\?3CVxg0QosP1R*}n6:+w{9sG$d\?$39q)e)aMPs@!{9tX2y6DL3:4\?72OF.{L4{yl/)806YWOZ^/I@N*6j2)Xh:-%8!4"
    "\?$46Q@R=fL9q54>^G{Aq7q5+o(d$Te9R/*-6x(Ue68%AX(e:rM5-x@f&MF!1\?$)uL}4<nH5E^Ia!-V$V[n\?toj]b6M01]E<1[qM@@%]!m1{1n@@%wA0W/3Bd"
    "}t*H2@qDM>3jD\?g01w]f@}$sO0%P8c@}#<42iC{)0swV2@RV(/0S-/h2kO+@00-Af0@A9a[bz!K:]R.y2j}AF0SSt(1oFk)1OR&$}$Qc)6Bsjm@}cl[3JQea"
    "1rNjv0r@x3}tQG$0%eYg3h>7=1nz]#1{]>T0T]4$0S&J63+>&<i$GZA\?PBH\?(iFvSHDK#m}Rh8Ld*=)l4a@cGgdiu9>>J)n]cJllXEY>JP+Q2K%k:>1&rdO3"
    "@26tA(^bN]64Eyh<($m*FE6Nw>*/nx3=q-A6nb]AbKS+U0<*(j0x3wXd^dSs82.bn4wCGE)9M1D6!MpT+M^+o]w{^Ja)y-KZv\?3//#RZVg.]Nd]F4M/Y@gw&"
    "2Ob!Q4^=4&]ZN^$jPa%Y@SzJ>@R^061pbhc@q^lc(\?QLR1O7%Y4*Bcz}1By.1omx-}2{DZ2k!P21]nz@0rik^%mLBQ[+-dX=JT)+}$fT!\?V-OG}$*l&3\?:-3"
    "}WJoc0tLtl2Mbqt5eUPS@r-D2@Sii2}WiDs}2Qp0]bDD%{5T4I4/DxU2j}=V0rJl1@@$jL1Oq=+3I:Ga(JEm\?{5+x}1O}](@cG8njQ7(X@RVS)>Mg!^%nz}F"
    "{Y7o#}V###[+@Q.[=F::{5+6>0}2{<1]!G8}2QQb1o31o1n@u9@@8=)0%5r9}t*H0}2QfZ{X:*<022tg{w:{-)/rv/2)7Fm0@q#$0@Jl4<]tCY%nq%21n8r-"
    "@R=P{1]we/}#W3d5cq!m{6HY[@R#@%1onb@00rh+0@A2#1}Q!h]-a%$@}VV[3fNwbjO/D=76i1W\?VzNT.59Q=3*v%g]9$={@#H}X0{yM%\?%J62{YPvH(IG}["
    "5hK7:1@VOo(=GC3@{[Rq64laP1%9Hy820*#}VDS/@s1nw@}DfS3h$Ll0{yEq3\?0rZ3<Dwe0q3DJ@}AYd00SXI2H%.60}T\?i}xaO[{73CW0Sft9*7aei]zW8r"
    "@RtNT0sdxK!C/jA1}7qf0%]Cc)=my!j]a)=@SH%P2)Z+p{3@a+3HvD$1NL%*@R((#02$.o2gS\?H)/\?UL{Y*-{>kZ8j1[Y$p]-j<{{Xj%=}#=V[0rS001QsF!"
    "3JZv$@R#/(1{T(7@P<)@@]<j^2k[J41PEPZ@pfT^}U{{b1pKVb[:u}^0{]NU].XN&0TMyB(&=CE4e6=2]znHU)GryA1onM30@r5}0S8&*}#DS/53Ra6jqVlS"
    "8u>V}jphh(1T<oM@$k!l{qEWz4+89N0qcf*}pU(1&0eVO&Sghs*YmxP59A#\?]Z6)G5=r%F[c!%l@T{mF1RZ+f)\?th*{tctN3fHTW1]/U-0nlil0[D/E}uLZ)"
    "0sfUI}2QWg(l+dk4dqQp(e}0T[b&4p3KM$f1o>7R}0!qb1R*dO(fc>i6Bku[{4X<r7xTQb5b+zy]F*7Ejok3N4h[Bh@OELK0Te=d0SJr41]4=R1p++i@%XMq"
    "{3.FD2l$C10sO(8@}(P(@@A054E%&969Xjx@qMQ1@RC/g1Pv#[0r0kY(H$kc2myj&4GuIh@0r8{(&N-]1]!X]}sz.!2mZ+d{WXEQ3&yno@@zrO00-D00qu3&"
    "].P3K1QUtu3i{[x{t)53@osdY1PdSZ(*K&I4*07o2mz\?(jN.}OF-2(75=JKo/W{o23Nk&W5=JBX[Eu9R>[p#va]H{l6Bj)L@sn]l].X*(cefdt{8&O:1q7z9"
    "044c<)+=K\?6auzH*Y)^B2l2keZ#dlJ7OHde@Rj!LZ5E<s\?tOt@}>:+H=&%*s@ON8h+n5K)@TeVN^U[.&=&>md>m]xTc/zSu-M!95<]tCSZzcUB49[!+03Ht2"
    "^g^2&{[<NpjPt=-0$1$w3j3.f}uE=3.TBGs@r]q7@RD&#O:DvY@$2eh[+r1)0qMP[@@zoP2ke5W}UZN34eG#y4gh}u@r-X4}sB}41C4*18$<7e0%F&G1onGb"
    "Unv9i2l>Uc4F*e41p<>[3ig2m{V.>P!cD&)5b=w22[u)S0tj@9]=AYa%nq)u1onGc)becG4/lh34d\?^PM\?}oV0SAu14KF[qjPLVu0$27@2kd()}V2QH00&l6"
    "}Vc9W(&vp&\?$viV1POD80SAc2@R=S{0@-020Sr2{1ow(i@TeVf0S@r6}#uOB@S0640oTaX2[3.j{XUd200Ao8{w$A@0TeU%1]F]5}#BIo2MaSb@rq#@1P5l8"
    "0@I#[{YHE61Qr]>0px<!}$Zr@{5C9+@@q)60@Sb]3i/I]0r&u22la]g1>yiujNraV{5:KW@Sr]k0V7230Tef7@{{r)2lak&0S-/c@@IJ-{Z3D/]AkP*0%X=3"
    "Iu@5v][02B}s+6U<q4j\?@@il7}SNO+an=a)}$gj25b=!z1POtP0seS8@}#0Z{Ypc<5^FTJ]9tN90ts>$].wB>@QPZ.{XC0>@qVZ10r.rQ6z&1u2Nq4z5FVnU"
    "@RDT6}s$3{0%OA90%e5*@-vy{jPks62lL210#7!o@SqV@@r&Db0@.Y*1O&e#PbEOh@qV^d0r&Dg@rAf2@S8x)2)72j@Sh%10$Lqf0Sh{5}#^2[%nrob1[q^4"
    "1[qG)2L!z[4GchfGAp&H1p20V0S-Gl3ihpP{ydf90@ir54fuqf2(TY0@SJk:7wE\?j}2yT41n-69@@Jf31oG5c@qM=%1N:Xo2l1J60S9Ge2M$mCjPk9X2(Iyp"
    "0@8M{<$Nkw%nrDw(kx*A3Jx&12lU558tBEa0q#=<3\?:F)1p1@g0S&Pc1n&ip2O3Rk0ShM&}TUm5{5b3!}3=Pk0SRMK[B>-C}2Z*c2MjJ12lKD34g0dr4/#.v"
    "@qM//<N.^L3i/[}}u=oY%m#r&0}l6M]zj4n0nWqO2OMam[at]y68-da}VW3h0r8^2{XjBN)4TBXj]k9X1ol:I1Qj+<6-W+a2mgno{YHV/0SRP=2lLkg0%OO<"
    "}#(G/}2fB\?@O^FJ3&Zwh{1/lr}198g@QR006-!pt@m.bD[DepT}s+x$2M%#b0S-uh1Qs2\?4G2z(0%v=\?@@-h$}2e$D0r@um@sw-g@}b<I@S6ze2MTEE1[\?6N"
    "2(at/0074m[9phV3&Wja)/9H=0S&2[1][tP1zq@EjobA-0Srr82laVk@@Sl31P58$[cG6(0T12x00rAa}tZS}(IZ\?/}V2BF1n8D&3I%-4@$b2H@@ql.0tbn7"
    "6:1Tw@quP[]-s%L}VU%U0}=D{1m=l>@@Sxi0@9903\?B(l3J6]d].!s=0snVi@R=3W@3\?Z.1onSh0TF>1@}#=%2(KF#{w<*J4g%6G@Rjeg.ufYu009o71[.{b"
    "3y9/fjoLuN@Q=\?M1M]::8@zFc2nuwk5D8>K@Silc(jKD}3\?&-K(KSe00ojHD[brpA>go2i9pS^@@pYiW&ofXk]-D3n(>6f$69xoy1Rza.3(GlH@urW63J{Xt"
    "4!G@p6ya(Qal->t65rSW02#aF*xr6c}SDO(\?UcBY7Uo6e2OC5/80J4KaM&:Y<]BA<4HzK&@Tn]a-[YXz@@./#]\?@PGjOnjF@QPZ\?)GTi)]B\?364B+d.2K@8K"
    "@{{cC2N7k13jEmi[cdb7)/-yw].XI3)GiQ$@p*K9{#z$I4cvZ36A!Qu1><aT}2{IA}uDcT}uuiR0t2th1]N9^2K8c{3j)aj>lg[[>(}7+1qgj]}0ed^1NTLw"
    "2Mj]0{3qRK@RlrE4=!b:7839s0]J+p5DzUe%n98^1ow(A@}u-W*LPLrjO]l{@q=-!:+xM]@q^h)&r]t^@SJ5(2Ma/b>>UtV@%eMb1]XY60%nx50q((]0SZEF"
    "0S@Mf]63hG0r-x7]-t5y@1O-f1N2B<0Sh((@Sr6i+ld}v>(z4*0T5tB0@q/-5D}L23<)7t}t]dA2(K@93&Qb8[=/0[0SqM}@W[P*{X1K*0S&x2@@TC51POL}"
    "2mgz701eI&2lT(52XaE4joLM++(E.A\?#jF+(LO1G@qu%q5afo91Or5}4IZ2g&s)g74i0s<<@a%D3h<Yl<q^:)2\?G4+@P<d}1]F6d@)QoBF.!X*5D9*s]7@m^"
    "@Sj.F[tDA43(Oss@$tb70qa!u>jF=.5FU965E[i{1o{z23i{\?M{yd2>@RNJr4dQv\?4^8mE}RY^l]yT*z^g3JW2<qM#]A%vN5BG]h{2H>7jN9Bx)zfty1>@>Q"
    "@q=MW}T::=2l9JK4/uOC5d/3T}T+a9@})iu{Zvcm]zX.0{X<*YT)VP2*XR*6@@S65}Tido3e1$-[b@m/)HX<S}%:N5(\?gbD@%Xq)@TX#0@}N390UbI42Nghb"
    ">[T@{5b2=g&SF/^10WTr-4ZJ!}tzn$4F{eb}26)}2ltnJ2)Q.o2>m((F=k]N{Y*P>95r(pjo(.@@PsKJ8pODV00iP3{6y%g1{<.@@@Ac00tUCG@r&Jo4FQgL"
    "4Hrav5Di1r}W0k&>kD7(\?V:z!0%]qp]Z\?f}}W9xi{XsH^0r@G2@%5wYS00a$@%X&02ltzb@RW64{YP<J5+u28]BR645!tiQ0SSJd4gr1n}S^kE0r-w)}#33-"
    "}UZl<0r&SE1PX-i}32QO&@e#M5bU!)4-g%:j]toN0TX![0ri01}#v8L1QL2g0SzZ3\?U^MQ(\?gtA000cr1PwO]@TF@93&YPe1nRV]6:1Hq}uNf7{5Um1}u^e["
    "@@J/r@px{V1}7nf1PXYk{4!:[FDl5R00@L{1n-t([bSB^2>[D6}2*%71OI*0{waN<3&P@9>LC8-%mMo[[+&s^0009200&Mb0S-A51pSGE}1sH+4fcUm2Ljgu"
    "jP1)T4fJ{s(imj=@tkns4b}m80>n7o0%nLP3hj=h{@ATq)G{Y1-qHew}2P{X5E5KB{R=Xa(H$$X>/iL]0%FM77wn7o-rb/9]Y^s\?@U$hc@pHWg2iL<O7B5IM"
    "9Vr]t[z:PG{XUrQ4\?5p72(4d12\?^t$(IpC>5B]f4[-UJj2iDE7}V5p]/9h#W2PS@-&{M%x3KmgA2N*WS-#7&djoLJ+2kFf50@@G4[^BWY0]0IS(eI(]0#d*A"
    "2Ng.j@S9&k)GAp*^/l#o1QUhn5BY=20S.%c00J-f3\?-DQ3jNmA2M:G0}#VPx@qlr>0RcP}0@&J30}tKe}3uS]4C\?r[@p*D@[a}Or%lG*Z{vXE!1oO6p01]Cn"
    "0@z]+}$*D/]-$D+].{2:1qRQL)!#$M(iNs&@qc&!4G#>z{eDl4jrh9o=IC#O<oaPL]:S2N(\?Z]M{bAM5a$jci74>9i1o[05fz+\?J5F1gy9]4<)5/Ief4C6zH"
    "b}YYj2d%En@)AtB*{t<S0ta}})zY[k5-&=*0&4gq%m#6T}P!^0\?246p><l*r1pa\?&/Y}R<6aJL59[H7w\?)i5c<qdn02>dJ:>lZ><Yp!cC2!z8UFDMRN%ka+g"
    "4h]E480pn$@2iTnj]LiT2k!n#}Vd95@%P8j1o5id1[qJ(@T5.#1QtIi}TT{^}#(u^(IZ\?R0SSc5(\?QR.0!G}B0q==[{v5aY0165n0t2kf@@qSP0@-r61OI]@"
    "0Sh==1o!#62MrcN3Klzd@RlD}[C-N:2(sr20q()70{PyY}Vl^11owxe}todN}#mlg@R)D22(aDc0r-q)000b)1{aY01]XY00o)SBjqeU82hn+T9R67c03&AS"
    "4gJpt4IxIa7yf761Nvf76:TB6(&3qG&RskD1r^BnKL.X{]TaF#]XQXXVJHyk80\?wh674jPXC#Vu[ELgJ9qy-n6.]U@@Sq9R2mxh=1p1#o)I3ci000Gj9$6pO"
    "[=v\?!{2kg073=nB0{h9q7#KSN]4SW91p+kCumu3(3<WgC]y8Z&3H^6P3]W$v(*14ujOX^.@1n-J-S^{#dde5$8w\?on@M2lh[C84}]-:y=77ZYH>lIm({v^[B"
    "/8cu*<j1wlckDj(00/yP)8Xi03d!$MX!(iT1RIEL&oq6r6vCp{1Ti@[)&Zf/2OcH}886X9\?RXLk0n]czd/F[$cdI5#{4A$a}W@xL)gpch]2{X:=Ice}\?u<es"
    "\?W\?i=3f6X990kNY6B<-^7x+9w<rW.bj]s{A}ssTQ&%kU-}2p}S}W0xf1P4=<)Fv7=Wd%!p}VcG@2k4@#2NZzd0@&G81N#]#6BK%A]6(>Y@%eSb}tHi<@q3Z9"
    "]AayA{8J#i@q(=B}2yA{0%w/91]5xf2Mj$)@%w&c1Oz({(gtbM1N#f}@qlP{3JH8k1qgXv@il4(@RMJ%00i2@@Sif6}Vc)21{jM1)EH5+%lfN/3nB(]jP20Y"
    "1nrcd1{TwL}ulG@0@J0g@]:R30$rxG1]XP1[cnNP0Tm]&000l97x9}j)e)d/{RDHZ}0c#2{u-Z)69evx4E0/%{5Tv8]A1mX0$LIl1onw<1Qaq*4/MIk}#Mr&"
    "3iZha0080V0snrd{6}9b{U$o5@RB+g0@zUx5Dx&R1l{l&4G\?)-3*l:D5cOr$\?#KhT2(<D54F/>C2w]&djIfkA.36Jp5FjHi4s#Uc(Y7]W(h{F}]-CVY}YjXW"
    ")i\?Sg@ikze009q^*XZmk*@e4A)b[]E}%$GV@oC:y>(&Nf@sQsp}3d@:}uM&Q<MrJ]+j%oI1FVaZ[DxK9%nq#94H7$-{T6MR\?Vj$\?X.PZU&Q$gP@qcC${7KFn"
    "[CBuP}w7a\?4b.=}{$A>0a<*J18t-+10Wl[p4^TG4f9yssjPtV+}viw(].FQ*}W@gC]85d<@T!k/@r&&fNBIAw}Wrlb2fv2d{0&<v1]OAh0qV9v5=imE{t#I:"
    "2l$28)e3T]2)oe>{)03l}7.:G5=z.:)/9y%0qM=@2lta#3hTL^8US7-2(+2l0UjU)1Pnre@S8G%=<P)J3Lqtb%1<YG)!cR!]zFTU[BV>X1Q1[$4ibMK&S!&q"
    "3iGL&5fa3Gjp\?q>)gnES0T[@l1qync1q*[D0T5FN]9bc%1Pmlx@qMP)}s0.L7z42L1o!A98#G^e%m2)=]7AdY@%oIj)gw<<[51}#4*zwJ{X>p0&^\?gp@Q{{c"
    "4D^F*2l%w<]8]3!2)*Xj@S&rc}tgZs@{Q3.@@90f1[\?=%9sh#O[Cr%)}$Z*f5DI<[!H.pn3iZLw{5>cw%nR@Z@0-#o0\?t-QjobAW8xli-84u(F}oZd*W=D(\?"
    "{tT#LFDt@B3m1^H1oX#x7zWIL2\?op-3iYAs0Tf)775[am[B3Jsa[#3G1$f6t]:PC80PB!<3l5Qb(gz*%!.Qi1gsMSs1QtRG5:Z8z6uQpx<K<!n^Jbv*}s1X0"
    "3/QJKOZW#T1m$<<:PHu1[yc$A&oPQs&04%x[:V\?I6ycb4:mGuM5:5i&4jiV!jp\?J3(>5j<cKZR2aSqr&)!mno2iMf(0$>.2-SlnG8T4Cg{x*=-2(1I)4H0pJ"
    ")z[.*2)Z2h6A3+L*YWDN}6cdp1o629FERF!@tDK>4H%iy0]1)#(gLI&0p]KE]ayZa/4*xc@L!%z2ezF:TNMtZ&rxO}6y^6M}%kE9d\?@x%2jgvF{s{C<<>{py"
    "}#bZt{ziw{4I4[8{vXZ)<GvAQjp8w:)iv]89%p<(:(AEc{%/xF57gNz4ESfK0Q6aI%nRS:iTc8r4IGAZ}}.5B0uIsv&T=&Q1OScgamxo<ecK]9\?0Bv^5D-xj"
    "4>4J273JTh4(%U70}tHp[d=/z@$URl1P/.7{51K}/6%ts0mJp){a=%o)fJ4[@$<BQ)^Yx70W!9Q8Vy-R(Y.b:4H&dv^Df)l(Hs/F{6Y%&{&5+WjKTVR{#3#r"
    "3K53}f.t$^2MB-w3ha#17j=Ip>i&M[@rL%<>M*eY0mgV7b)Tbt{w>:3<(bKud*ty&8v4hM3M5%k[:N+i*yBbh77\?ib8uh+})j^rF>\?U!BalR!\?(j<Tj@sxpo"
    "0T*d}@#{C])^e)3!fTeL597+e.uhgY1R.sga@Jcv@}155>*f.qda2yu}#2D!8#nk5!8Qov5.lcEjN@EK1OHZX[aDtN}4<Ou2mo@g1M/vx5dn*D2m*[v{Y*]]"
    "]9%pF<lcsq@riGs2jzlZ2GDPI}@^>x)=0)r2LYnv5:f=e)G9cm5cJju8uxNp1nAn$@}3G}6tBbA8X7l)%nAA5)/AKGb)0Yn5BYeP1NTvX0q#{o{6*Q40RCaJ"
    ")DU$x3I:bW58FJn0V7[i3I:I:67(t39%.2(>rqnijO!WD3m#5L@OEwJ}u^@g{z\?Y{[=F7V\?1\?rX{TorK00&c80l>.E01]km[COdU2LvD>)/-Q%]yJ7(2)ZLt"
    "(Hab$@09o74dah>84VJt4fc>V!=2KVFEhLU@%n@h2j\?@X@pVGR4/MOz{Yy0:)GAgJ@p6pS2>[3a{A^=(@1nC[{4OT\?0RuJ[1RH.G1Q1#25!:<j<m-9K{u}RG"
    "6LPk(joj)G3<Npt}tZ:0]8W[A{X>6{@QY:FF+GuK2(B31}s<^V@0JYm}%sm:3h-#\?@$DoJ3<3.u@RuA}2M+8e]:pG#]zx3[3\?%&b[C07J0R(T4]zX<!@%Pho"
    "2Ngwi2)6#a@r9b$1n@oa\?W5MV}#E9j3hjF@[B#$>@@zD]].OZ<}W0if}3u%j}$Q:34dBM02M:#g)gxu@}XN9D3N{m+jOP02)ahvf4d&MS1R.EE}#ug40$aX<"
    "=j73J6+7T3]B7PL{x}w)3&gk33<Nay}v%BK<Rngi(C9lm6Z(5g\?@rJ18vT%{5CL-f4Jtlk@@iJpZx0:69%@2u}U/)^1OqY&&$dvp1nq{c]8NOD@@rcL]Z@Kg"
    "1onDk(Egp{{WNzt[bh!C1(xp&6...9@]sv-1Of!F1]!2-@sD1a4yZ*vjqdIT3j2(27xC-t3\?%tJ719c]2LXz-{Xj@}{C(gInfOqEF+OM!:]o@(0RDEq6ZL.]"
    "m6l7.bL.-J]6bPB(gum+5\?aXQ<{Hkd)GK<z}#3g4]Zh}w].!j/2=C]s}YDTy[-P/[3&Rsbh#0BE/Xmhp[-PAA@s!=o]D:R<2PO<v:qKZQ1RoJ2^g!btbPLO6"
    "5:n=-{Z^I}bI&R140MX-jq^VaaP-z*0s4r>({Qyj+\?i&$*:st${)@hQl\?{SY@qiX)/4pe#c\?V%l3\?bCF([c-u!^HZI86Gh25BGob/c{XfP:}L3@$Dx5f53+&"
    "4yMf5&*mpK1nI-gSR=#b(e-xN&pc9r<k{L0\?@0U%[^CQ4&q:qx@&Vhj1L%!e+hLCtkmnIP0OvnS}mc%>=JK%S63}x&[c/saLMpvv&LGy)jG/Z}*wc9}>>Ve0"
    "4f+!X4*\?p%%1Bv+4^%wY0TY\?78VGNa]-Ur%9r=e[[GO1<Od2I@@o1Lg>lH[}*4+aE{Qw#P0R2)*1pC[^{X:B{3h$[k4cB@61n&b%XBeF^<(Vyq(DP\?w[CSXc"
    "ap5m\?@{[2i4a0HO5^nmp4IY3M)!fb*}4@xN[9}FyjyPcv>^I0m0P:SU]+cJ]!c=Eb3FqG5jp7AX2(%z]}#MuO@0iob1{s/h@Q6^)0S\?]w0%n(c].!:V2lBC&"
    "0Td@)@Pi}:1O&o2}$6B*%mciW4Eily0snPd1O@c1)e($U}$[Oy)HON:2NgXj1np*A3i5#=@Tni0@@ios]z:Nu0Ua&e@o>9#%lX>q]:pM{4/u6W68xJI0qD-]"
    "1{gg{{uIFL1P%tJ@@8&{3i7q36-vQ\?@0]VTjqE2<9rtg40U>W{&05PW3<DRy>**EV0Rjdw2L^uE]8FQR[d2D!)*X^]1]Fl94e]u9@}M081LBBH}$7J]}t\?ll"
    "(\?x(F/8bgo@p/dB2NQ4#0t(Hz}T:d)aNF4Z1(/vK\?XVtn%nQdc0UCbr0sXG41Q9cG1Md66(jB-!9pqqK8TdOx6=R.L2H0zW4\?wHI}tHA\?3J{E/0%mBe]+V09"
    "jOPA!&{^D.]0/wi{-]}m5=}we2M%5>2IG7P1p+nn]92f@[c]lq0Sq@(4f(Ra=*.5y]:7K4!B-$O2(aO#@nNZb}%Uy47XE$t2l0&Z@RMrD>k0G>2MB-g0S-zJ"
    "1o5@l{X<K61lZv7{sOV^0}E6B0r8P=@QP*([7E:q[DW>R(*TSM0%5l50Tx5n1ptCC2kFR}<P22-]:pY>1amLmjo#b6[8$zY4cWJ(0V.u:{4W.H2k(1v)/ig*"
    "02L}x)frUo*7JYH2M<.!3&[@R)f=.l46HuA>F4&x&065/1pjI](Gn3C1m3xx@pY^^1m#Wb<QI<!Z5Xuj3Lrvu1{a@B3Kl+d1QL}s0{r2V6zSQY&qSMF1PdTr"
    "/-o!j@n(I%68-6<1PeM23\?TI)2M<@b1P)YmFE\?X>5BGC#cm\?H]jQ7=+3&]!%%nJ8^5=.!e1m(G)%iH/s00z/>1]wn${4[KX0rh]\?}V0@e29]k>0qDP[1OIHd"
    "}UG*U2K.S=2N7qk2j8AY(&N1:}#ll^@TeJ61p+Fj1Q0=+1{jF#2ljk{@@Al7}t=9v00Sfa0@I(#00q-I}v8=*1PwC!@}#M(0Sq%4}Vb%-1O8-]1omr.3&*n8"
    "0@-M31Q1><}F+]ijp7&}{ZlY@0$LOl3<c!75/76r2lU8h)GS*.FFX.00r@@C{4vUy]znN\?00Iu/@@RiU{22&C5c8\?t1Pwq[@qWh^0}M%9blvxn>kbqY1o!L("
    "836%fS@@w31mDkx3jc!e/A\?M-1@4dA1oO}#5aped1]v/>1PFn3@$(aH0)Hik[=w{O4Gu}p0r%bj{ZD=(1l/^V4F{512g0UU3@{dMj]U-U@rrz&@Jk.(}Vc=("
    "}3bTU@qV:0}#u{1]x#+Q{w$j31m#=[1o!/c2(+1$1PO5>1Qkhl[cGa32)yhd{Yp5s01nA2}Uy9(1{B]91n&la^/GWQ>(\?m<0rAC#1nS5{}tG*:2MjSc@Sirc"
    "0@&c12MjF@3JH88[B(+:@SAr4{X>m20rr911]5691nSl01{.Bi0Sz(@2ljP4}z.cdjK&=[0}c3.8#]:j6Aw1q1mcpdbMdP6@%]tm3dF8E)/$0t0sGd<}%TE."
    "ZZoh(3HVdDZ6%=8]:RYh@%Ghf5dwd8<>*Zo=J1Md{7Wk(bOdP{1lX!B4cVTx}#^Ap@S8/@=f>X[&%2&f%l/^]2hoB%5^uBt3=A1E0rhV)]6#wL}R}Cm9VUE-"
    "7WAmq}%:W+}tyu%5-B96}um]h{G^l-jO5sL1Oh6]5=TG@(*>8J@r-@q5\?0IQ%cTGn4*rNr[+&7Y0r-t!0@S652(jz=(>P=$@[W2c4b}fI4=uA](($V[F=bJ8"
    "[cxgX}2Qr)]qp*I4hfuV4fWu]]2gtu>lzb30swY2]7I}!\?@<!G0pP*B>n>nV2(9fP!+]aG@oKZQ7vI8o8V)!N6Ae4i{wUB51n-A5>lQqZ2{4$51}d&{jO]+x"
    "}v9CA0RV0Z]-bWl[DXd:%m=x)&R%h.{X+j1]91<*0r9oh{2V8G3h$eg2)Z%J3jl$z%gOQq1paX{@qc3n@riMh}pLuF3kial)frKX>M}s}}ul##1on+R2kOO$"
    ">Mpkz3&Qnp@sd/*)F=+w2LN@>}uu/<(hGAy00rC]]8wN&2kXb@68I>K3&yqg4h2rt)/0sH1omcI@uGgNjPk/}4EBb}}Rp=E}VuG}@SI#@}VNG61m#]@1P/OO"
    "4\?5Hc2Npee{yc%69rVJQ}#d9g}Vu{60TeDa2Ms.}{t+Z{4G328<nwuE{Y}5#3hUE#4Eae%3jle11[id[3&GO&1{S&W0q3x^%m=Q1}Uyj74fcbf]=<nm2n)HJ"
    "@}#T0)^Qel@0%ks@q4cy3h%R{4!w&=4fcI83HdJ>8VL}kjpg]@[97t=00:+D4*8[z0S-t$aIv-9@0A5C1P{yG%k17N!=UEL:[CU.4gIL37Y+6/2G1}b3f7)M"
    "(*<e!@tb[x4zS8C!CY510qEr2{q(Q\?1lIfH3jlH.<OXG:(>!:(*pN\?6(Kkro7XD}J{4f7q{qbX.[^>(\?\?VKGt)!3\?T>[:{E1OAJl]eNVL(*SN2{u.\?y4ghw7"
    "0%PLj}Oge^jPUPW@r9xb2(aJ3@@@5<@qD-#@p*r<0Sr5[FE@1^0ri63}11Q{{v[K/%n8&$4ex&a0S@8}1pb25@qMV%@S&Y70RuZ0}U*S)1oFG7@SAAb3&6=c"
    "@R=S}@@Jx90r-t$009r8}UHu>{Yg^90SA60}ty6&1Pnt[5DR!l0qco$0S-u0}VlV[2MLCj}VlP}2M25i00-ua}T>3(2BOx<jPT)q{.@@w0{xQZ{xHY:)frj*"
    "}VcAS>(I4*].wQ#@qDN7@{G:S0%!&c6Adm@1muT7=J0$D[DYg62)y2j}33AW@@rch}2yP\?0TFYp1][-k)!LJuFDMbR0T5&b1OR]^1otI93h$2e]78kQ@Q{l]"
    "}s%)+@@&lb{5<p.{YQ)1}2Z&)0@-o4%m(J>%nJl82kcpk)EHkU0R#%1361n#joM2={ZN915DI}k016O4@{6W<0r&\?/SS9eb0QfyO}0[d-}#m31}VVV(]zDAo"
    "}VD0N3JHkn@se-n3iZ.x}tZ6-{xpJ>{w+B0@%O&5%mV6<{ZEGmF+66q2(1k$2M0#/)/Sq9@%5G4)\?tP{{ycS>0@-A23jE4R]aqrm0sfh85b#XJ}Uf<.2N{s/"
    "@@Jh!3IJ#@[+-4K1SvmA2(cJLjOn%O7Y0nf01]XnN!tXM}w]\?H3I0==2Oc.H2jqPV{U2Yy4G=v&[aN1#1{Uni0%!t=1oWu2}vymo2){}p}3V*f5F1%{-0kSp"
    "}3kmg[=/o]@qDcZ0r:Ib)G@^P3ix(5[Ag8J&%a=I(H$5l]7A[5]!Nc5}2F}S1P[.&0Q{K9Wflc>YA3NK(j0N<@qs4UFD#nW@}#-b3h$qJ]i02yjMs<90]%E&"
    "1%i^+0Ky$(bnQU70PB<K5/6M:5-\?%v3\?T.Z]-J>D)e(T)/awe+4\?XWn*OBS+0lV!.=DEBd>>>5+5-AuX0V/JuSOdpS1[yyw@nO4{2ikEU9pzIY6B$xE)Gj3-"
    "{Ze.\?[D)[i}@52u)6ax@{W!t7(bq:9\?%t)@*VO*m3gYUv>&pO90@Jn}}tpGV3<)vJ]xDv$}*b(ajsf!b5cH[<1R&cD[EkWk5CM1@)4:M){XT)q1pj@3+hjoo"
    "2>WP!/-I0V@S@fa8W2)s3H#4x{xHGV1Pv=\?&%=w81jWng}tPXd}}i4v-}Of<4^9#]0$LEoa06wn%nRo<@]KB(0@z)5@t2[^4f[G)2(KGb>>bh(ci1L<\?0ty!"
    ">mWL0]8]FMUoakv0Sz=Z2<R0.@R#-&8vlrE)wo-1jLxPm0[b=>2=)U6<{@%F%jWIL]ze20}t\?ci@$=f#}%+9M{$L]E{YH:d{xzi64D40M4GlndZzu!w0SR/%"
    "5EPkk}uciH4IYlo4EJN40}M.V\?vfLQ]Abd7[Bv9v9]E{VbN@Dy1m=9R<r>07<PC!=)F34/0u7zr66Y-3(KJB$85}So2n1-y73bhT@SSSe4DEGU*7$4\?[DF%U"
    "8:MeIjtMmk]y>fY0TEPO2Kzc(4ePt[}2y.W}$Re)3KC}]fc&1gb)T\?R1T0I0[-PL$]DSR[8Id8u5dXvn+qfCq*@/:q2P$8!^fZc)bK<Oo}3[8k&k>@{aOBXi"
    "V.u<{Z$^Jd)EfIQ@t$Uq9P4b30[bNx@1]K#5G@a)Z0(o^[C$e=6-LYt^8utFfsRFC.0g)s9Mv%G@2Vd9-]c:(@eBvqjpZ-=]:^n#(&X7a&LaZ.1M{3(}s202"
    "0Rb-)59{%([!ikO2jyQm3jvWR^e:N:)-DHG[D6G>0qD@.&o}A][9*O^3F>Q:3kAHr*V5w0+i+>\?@]1Wk0pe>-{.%Rx}R}$!2h{Eg{Z!ec)/0Nz\?uCLB}#t:q"
    "2^%^o{u@y^(I6(L}%9FT6z\?gAFI3xV[c]^i2kvDR1{Uwm5D&Qk}Sx\?-jOeSR4+pdF5teB-)g{(0&$I+y(<JC3{2uF/=&t={4Hq.7}6C}p2<.r.<PM7$0{9IZ"
    "@S-cF3Isc53I$bv[/3A*]ZKc]1{thw@$j!$(Ha(t&@F&L2cW*#>)D@31[0nl0o<Ww5vI-P*8Y\?44!xk%%l4q^5xsXX]ZrO4\?$miB1lov<{wBZ\?0[Ez!4/Wpm"
    "7:fdC+7fMm>[sf/3)JLxjM>Yu]-@Xw}3n8r^dti:54qDM>mX0W73bhW3&s6lh.=SS].C{774}E&^>eoi6W+Ur\?s*<V}V3=V74ApO)zDTHdARJec>>B9(<KS@"
    "e+b.b2f[RS12d5p:Pr:W]4-mj)2@EV+LAN\?@U1hR0Oo%G]sD5{4fud+hvGTyR#Lz>^yQvG5yJl.bh9R5@S-/B{UbfjdcZgiKMmLm/$WrAjpR5M4\?wdj/7oNu"
    "@$t\?L0Q*v.2<.{7]4a9f\?u1/30}l6p[:VLD1}Z.I1Ohaa)etca&m!.s{5JLM{zTky@nvC7<pqI5SsBEX1MY{)<Lpsa{w0FH><wxb8oqc:@%/Rn(&#kH(br=H"
    "<MCK#)bwfT@0h(L{SI^]]y.\?$02tB#((DMC04cFq<PLhz9qo3]00-bU00zY:@0&eC<8<U:jotoJ@1w/4]9B*.@qvzd}uVS}%mVfPFD#nV2O4v^@qVx=]-b9%"
    "@Sri3@@IG$}t61d{ymJq)!uO^{Y{x>}Wh][@TnG60@@nX1Q>ng0Tn=h[=ey\?0@qM<3g[wR3J{Up0}MA+0@&l60tK8Y2M$zu}1:KW0%5981@Wjv{3hFz}2yV)"
    "2Ms(a00Az>}#W362my[h(i3>]]-$0/3Ic#$joVT8@r&I]a&a6Q@}D=Y@%ne%}V=M/\?V9(Z/7PyD0rq{2@@92^0SS5]1n})53gEA.6:1Tv}uc<6@0Ji9@qVP5"
    "00rxf}05H=4g-mm1ow-Z\?2E6RF=c5T1nJop2kOt=1n!tS2<zr\?{Y{l{@}M6+[+I>=2)ZI9{:2qo10mHH}U*i#}$*r)00&Y00S8&(4f%9l3\?sh%4H9mx9W#N."
    "jOdCw}YDv!2MCLX@%/B)>(Qn/}W<ee{vvOE4B+Hd/v6%I2jio7d<s(Sg7rgV%ne}\?dddDx(l36h2kmGv[fXOr{xNR{}siT0pa=LC0RWoA]y-Xb@rA683gvJO"
    "4gs0e3Ir@)5E{9B+!J#FPA1Dh\?qvI@2jpn8&]#Nv2[tI=[&c)s1{jlb1)KEt@}M^m1WRF@!fBR[7:FIQP57-mjoqcf[gs+j5^m>7@r9f976JUv)f%i-F=t(L"
    "]y-<!}T%7J3\?%Mw59<k]OCwMN2(9cw(i4y]\?4Zs)2)Qeq[Ypt2]a7l&9q/<N01n+7\?tdx@>h^lW<n/-00{x*]1l*G^9{7rq]W<u=3iWb[]7&4P>&zTYMfX>>"
    "@Sd/\?&Xf9V3\?iW2}%::.^*=VI}1+mi5DSpC(&3n)gB7]\?b4b-2jpp=Z3\?@lT1]Gnz0s/ha1Oh<j1p+qS@L]rx4/^ZraQuP$67keT3&y51}tpA&}v@Se@$$O4"
    "0%F/c4&#bD[E+B2F=bSg@Sq#D1OqN2}>jk*1[H{^2]irY}U/W[5^e1i1qHUl{u&4T}QUMN2kwJ1}VMGc}2]Z+1Pnrh{vG142\?fm$7Y1<q]vs]<5^!u/1N(Qh"
    "@S\?i.]:7G*{93h@jos:f2iDcr5BYF<{7CGX@@}o+0Q(fBXDq{nFZ+FA0quM}0pGT^\?vPLZ&%0Ww3i7N=}sql8]URsp[!7-{]CD&M&sdHT{2k=B5HEXF\?u:qL"
    "]8]3<>&@rL1Svdd4Gu}XUm7)v(LfKc}uMJ]3f}%70qt{)5dOY*6a>Y:]+LQT1]/ng3<3BL%m)iE@qD<22k[=e2H&wE]SW0C6{Mv(jOnHU0@9f5\?VKeG:nC&r"
    "1{<R[}S5241r3w83>9*G0S\?(:6:AEA0SKz}}#c6<IVJCY[wj)J2L>*R(hQkHQw+n}4E13f&S6wd00eq:1Q2tt3NBus1o[@k3FD9#@Ol&j]Cv&G&%i/S@snu1"
    "{rzE#@OD$}2k^Yx[9Gw08UBm0@o%)y1P<>/55oj/@TkRo&]{S}2MJ3G2(:+:5ZaNjjpy{Q<rk3k)dH>[@r:ef[bqOS}3%EO=gIkj{zT8w}2xj!3h1[e5fyok"
    ">K!oE}X5!Z<mr(#00SF#}t/))]b)\?-}W<hv4^+Uq0$j}$(&[55^c{z{(JNQZ>LtX!0@Auec<!n^6D^L=}Yt5d9@+*N}t7o:(<<mB{42{92]J/D]zn7X]Z@B&"
    "1]XX$>>b2m4e+U-+m:EU<pg[%1[GCfj]tfT}#u-%0rq-$/9zJX0TnPfY.\?:i@q}JJ0%ew$}Vu@}}V3o.2lt811%0Nx1onx2@%wPC%m(Y>@@h-(1owV90p6^<"
    "0rJh#000322l+5f0}lK31POA22h/W<1Oz/{}UQf$}tRlg0@&h(@}VV#@$t5o@%wYf1n&4[@R(M>%m#*42Mq)E2Mj&c0r9M8+kRLC1PO(i2(:@018*c^jnegD"
    "3J{a}b{rK32UM#FR[a8B7n7#Se3QgXp[{q>dZ4{tZVNtj-QaA24d2Z)*t:c-(PUWP=h5:p:mdKY&%wJ)I\?e5x+[H#T\?VA}L3-AE{^+.\?D6.$Iu(!So&(li:$"
    "(]1Q&5.!MXPzc05cn0j7>!LVJOwjt=7ZfUp\?PUml^eFR-[7NdEZiW41Y{n]%b(n-D&$]Wo3Zl$Ob8Nu#jNqUhd<aGS}40]a8@Y/.66nl/599)N)=b/54orYD"
    "]aRbVUfb@1>[AZq!A>&/}uN=p)=aV$[6-&Z0QZcK]v$Aa2)[r45EuUgF:hTXa35].[.hWO5^E$He5\?eP9nzRu4GB-v@p{r}}[HUP6GuvKeXf)A/EbLo8o5cR"
    "/.(0<0S8s3e+P[s(<0$c8x/4=8RVD4@soR(0wirK<*KrmjppM<{.Jz}{y)&1ebfzZ0S&feli42u2]z!81Q3160py6[/7]:X&S3)!1)=m53<D1<@@Abi1rVW)"
    "1{$H{f-gwD2(A@]5c\?te[bL2)1o[q!2m]k}@Tnz$/9qcz5h1H&}RIUJ(&DnB&RAob1o[-l0Y6fE[!pE482%)5{wCp0P.53^}1T1P&0oY96-AQTLL>jU]At0X"
    "]5P]W0%BG8j]UM:>kLI:}tZ*1@%w=80q={3STeR00%FM70@@-000Au10SR@{0r&Ae{5T)<}uNfa}2{)44heyu1{$Lm}U*G%1onr1@qV@>1Pt[j{wBEQ@Ru*h"
    "0@&rc%nJf800Au4@@h(#}#V)7]85B[@@r96}uEok@%5Ma0qM]}}#u=]3&{za@ri5%0r--m1n!.YF=t=K@@}/!@Q{Y}0qDIKjpgcz3f*NZ<]UF!5bV>k02V1N"
    "0S&A23/q6^4f)Bg]8O4R]Zh+)2l$z95=8>a]-%QW>yA:D9}*lJ(\?-u=@T!]r@oA$/c</yC4fcFb]yhkc1QD4k2{60x{X%yW4HJWn@%Pwp&0o-0!.o%y2ilSX"
    "}si$3)+mjp)Hfa4)/pU}@]LozaS*1[1*)T/5cRLJ0%e>d4GMkq{523:0./6@jP+#p@0JV03Cllq}#l@>2km)1}tPaq@@Sx7F=l2+}#(^30}=S=0TF=b[CI>W"
    "0qV=#}Vv68{wC0U0%FD6{w+0A0TwSj2KID<4gJgi0S-62&QW0E]9CT10rJub2kXIR*XZ)y3h<]h@SJe@0SAf1{XjN*@QG\?0}WJYq1{U83}#v09@qcA)@qMS#"
    "0@h]#3i{>s}UZ9<000312T9:yjN-gy[c]0(@SR--]AkuE+I&me@r4Sz<]C.#XbtW}2Nyht{3&n10}(D(@@e@71)Vf>=J:TQ)eW<.@@@Af}Vk{R)/&y#)ec}("
    ">]6f[2MB@b\?3:hZ2KJf61O&A02Leb(]8]0Z3/Im0>kkL>@[^RH<OO/T{X$fI[EUc@>N)<}0@0fm@Qx<!4g0mQ000C\?3(n^vZ4.ak{@&jkdC.$!jOb^41%jr*"
    "0Q{f&}U6:>)e=+X@TnVk(*$C=}uM&=0rRD+@P%ZD2NQCc6.QeP1O\?fQ>LLz^3hsI&2(%=b}1<yE@rJu73<3Oh7yZlk]-1Cg(GnA.Qwax.)GAH:@pQ9^5CTue"
    "{YyE8)*aHi)fJ1.0q=#-Z3(gr3K<xF\?wl<@5Ck+#2LFx3>)^Kr(*+5-2(yB\?(<AW/00rl2@cn&njLex={ycl=/6hw[([n*X5:6!<[=pcmMD6B1>&ri^!9T*U"
    ")cBDg1QaO$^^]]x{BA6s5c0^)o:6Xc1pUCB[xTzV/Ebg>8v)p]2\?e5[(H:061$w:O*PYDx+)aW^]Zr*bJO9Q[ek(Sr>{z.J2y^$&puf+c(+iI@4GWZV2Kr-D"
    "1rV>i[Fho5ZUG1wc&T!.6x*nqbin*[2NA6X[6g.Jjp}IT1{$8i7z3MO&Tlj^Zs+E.8]y-Z]6(q=6E+%d&rP+#0W4k2gu!02jQGHp{6js)8rZ[]1TAsC{Uk&O"
    "@Rkd(%c--%4>#\?G]7JaT5c\?$0{2C=2aTDX42>OuI)\?3GWJo@.i0PK{Q76TcV!9uuG*zGR)[z9vEY=Pfw@dj2w2ne9L{t<A+lk*A->jYi6(C&tm4jRhA}U.ie"
    "=C!&XjQ\?t&2)6L%}uu=S3LiTC0SSc91OI&>@TetR2lC575b#Ia@r8==1Q1V50RVA}{x7x(0pGHv}1:{<0Rc*01{L2b0Srb$0%nh8@R$)I}uN35@%Fu32lsL@"
    "2M<S60s5w$@S-re[=vUo00Acd1n-910rRfQ1n\?S/2j}kZ2kN9T1[&fe0q+^C0OmzP2M/LT{X$u(00z&\?3i7e8>X#cpjO[)W@}^fb4*Qn(X+#Eg0R#*e01xb3"
    "2MSD{@r0o9@@j5l1}HC9*4^nL]8[^*!d.M)0Sh=T@Q6Q\?[F7g4@r0f2}0)\?O=\?)v4}sSFI4eob$1]!t-}V3y02L4&<4HRCd0%fbs1m2N[4ee<82l<f3@U$Oe"
    "}40uc0SJvP8t&g*3jl%}4dB5.6Dc]C3h\?Ep]9+Qd@pF[S0s5IR6nUMvjm@E1*wx$q<Q\?yT]9a%2+jb-<>)NvQYx7da\?@:OI.uffk8N7j+*oZzL1{$hS}QUix"
    "@0&us+%.N3[bg*a>LaWf]9%vO-qyhl8a+qk3g3}h>k+hJ@uSfDbM[A058wO{FE7SF\?L6Nf<QsoLPyA^+*v2Q+0U=4&@R^3v{S0#$&Tb.=@m}qy(/P/t>>UtK"
    "cN8m@<iuPZ\?>GrFiM+t#jrtg4/ClyK{2+ld%eB]U]!uB/6B$8$9tX4W5=Y=o&RRl#9P^yG2L5PmWF{:v^\?Ttw<#ih8}s1vF&sLL!cB@^G2Kp<d^HcJ}>/0-s"
    "[C%vh3Xw0T2LNAf(*t5+}mvgI@O4/5}SWbj}VvGsR%r\?9}6EcV5^4jz}{Lxj3HcKD@qDjd69Vp)&rgs<1@WA}[Hsg)}VXw]]SNS+@<$aYjMvHJ5A2kw2MU.q"
    "[bA^(%mMV[3&{2#2l:X\?0#JuR@@h%11oe60@qcAP1oXn$FEq>(3ju[n2M$2r\?sp7D@@@D91)Vv6}2QoK}ulG[5C)db4/2UJ0W4as]ZH(G@T/Cq\?U)([@PKH^"
    "4*@Wr]zvnI>jep72)ptf4grTN0p61v>lpF:}U}rmc)]MB@p{=#].npp@RCWK%m^8%0J2grjOnmR4fMGU3gNx^]AL6IamU74@RqGU(fn-W}q.kB@pGdN{@J^n"
    "0or>Q@}vn(0{o1s/-z(){sP&>1p>Cu[cXTL)E]ur%b3WD9Um]R<{7@9[E1^&].!)V]By)f4d<UFa}}DD8$TvR[cfWc]Y3h:]zeBO1Pm#<apduH(koZo{Yyx%"
    "4Ff]11}gUr7553{5EP/V(.:Ee{5jyR1wj/Tjpg-+0sX@i2)*:L8t1NG0pfT/3dxg{@0w8^1o52*]ABNU1POn$0S:8v]m^5A]yA1S01[}N{yD-U{yE(s1oOt%"
    "3iZhd2M%XQ1[-cb2NyFp@sd-=1[hN26-eyu2)yzb}W@Pb00N@i{Yy:20@8G*]bWqm3e:.H@rrD38379x}SXg!1p2Io5Zyy*5cZI([bipF0r8=+4eHno*A2jd"
    "jp7(:4^%@q{YI-tkmX2b%nI<p>oY@t1{hd53KvyC00r5<*Y)-S/93VG4!]G2{.rMg0ULcT@r}->2(K2!:{5B:]yz$N5Enfg}U64H]y8[+0TdD{2(+\?H2ibKY"
    "aVQzu1oGkn>j4ym[a#!#3L&<C06o#/}48S^2L[kN}%KBJ&rf2+8v=k*0V\?s5%nwO\?=*pVq@@H{K4d<+:2g.c&jM$q%gU{%#5C>xb/.===1N2<$*yaY0}t=7b"
    "5=S^D8vNFd*5G\?t<pBAscMqK.}U{)7]ze{&1QhTh2mZ\?G3e^6b0oq$:@R1Wc8%d}n4/(}F{qEKB2k[nOc>uZ#[W]87/y[[V4f$]3}R\?!R5G\?z!2M*Og{v]%0"
    "1Nloo)bBJDd(4ed9pAO9chw]4aOWUM{w+fw(Jkuv{$$(Fb>q$djqu]1}T1D!{%GiC)g6>Zi$7M&{X&dT{ZEG9&V!gp@0rJo1owSk(>Pr{}tgu>@)z[rcj*O4"
    "<[5(=3&G#l)iRl22[^)v4e7v!1sBe\?!DlPZVht7w9SD/y@00Dx5+(T$7X^d$0]KgR\?3:wT@@T8S@QYW-[D5Rc6A^z5[CzCx@QGR0(G!Gs^!Snj>kC&23(e)s"
    "6}B.13HDsP5vlWAjNQ(z&q#dC3kjo:4Ms1!629rm(Ip(6}3uw=7YkrOasNHc<rgue]6L#a5IIM7^CTyi2}+cm<S<Jv.gzBs]a{%nkjd$x2L!t.[Beav80pI&"
    "]WKO*3\?J^t(J)jR2[n6OY4^hZ0PVAf2N{NR)+$Fx0&hkF2kWlh&Qk>-\?tVZd)=%u+)!PK2871X.*$9>U/ztr<@sw5T\?tFO9Mgfe*jok&)<0Bx$@T*:X0UTM9"
    "1<3lP48rhX>[%=Q4J3bN*373T]yh\?Z[b8OBaqbFie^qyQ@RkvE(gj&J/:n5n{6gBe[pyDc4f+2+&S6#17psnw4:(Dz0@}=86BKH5]aHBa1PFcs{5a}1)ahBd"
    "1Sg8T%nh(%(IHFO)E}5b<kPb2+K32h2JV=W!(8B&92ED7{6Riw<PC>/b])jh02S66job(r2(T[{3j(!4}u)S6!9EnC}3co<}1Uf{}VVG{}2Q9>@@Aob0@8P{"
    "0077c3j=y>/A.(\?{4/420%wxd]A$c+]z!aR688L12m*Fq].w\?7]A%QV<P>!\?}3d67@pQJY4G>>\?2Ln9d[bAp.]4J*C008/+01w@e]b3S${6Ixd>(p.>@Rll\?"
    "1}H.D1P5L>4ePX)SdB*v]8!pV00WKsjPk\?V1}pv!dY&Nn]8X#%)Iba8{67S-UKAg[&o*H/>>%V!5.mMw@R=V%ebc{}8S\?ty4*%6O>Mgz<0V81O(\?H1B]A(ly"
    "@N\?nu6Gt:X0Sh{o:Phev@0K5l3KD.S9qynX.$nJV6xo9e0Vwok@RMo^4Errw2(:&q((1+5[dluf[:=$=0%OA42]13ZaN]H292/N&[b-j)0qWug8H]@8jOwm9"
    "4*IH{63H7J@rhP\?02T(b0T=)F\?#Kh(@o&RK[+@Q%}VErw}0NwE6-dks1}7m!]Bzuy[D]H43>Adf%k%0a@$Mmx]y:0@4>4wM>>$Dc>kth>7w!WI}2Z/x2k[Yf"
    "5C:$[1n&fc)/hqA0pQN0]Abj4<{7X<}%Ydp}X3pd3iZa<}1%Z(@r-Ye{Ypi>3(47}Y=ms\?{w%%-1YZr\?jtMZp=HTQsY+A6}3e#f<[Fgr/4HqM14vE7laR#/S"
    "9B^U2]ZGY\?*4${K@}$<W3I1ez03za91]E]M{v3qa>EO.sHAkTneD::i(glke}j3)y(E#Xo9t51(SnEE+RZO\?C2^17u]y:X^@PA>C59M*{9nc#85XNcp9SVTc"
    "Na-bS(!KO@X6^$T{=ftF.Bahf)CX&N@#Ri/F+{Lz]:hu*jr1$w:lj09/CLIb7y!*]]XmT$0S@cZ1OS5S2k[&6<PaS5FFn{/5^nQG2g@eL^gm2&0&9%i2m]$E"
    "Su.3n5ZnrtcmfVN1N#Nd[b9Cu<HkAR1H]@(3GgD#^!RR(+@byT9L.FD1n.C/4-vGP<p<7e>*GbIf5OC70qK>![B:-CYXn[W.Sp$P4K\?U)@Tvu<1]O}B67(>c"
    "QTo#j](WjvjNq1!%l{D@8%dC40$2Cr{x*Q1@@J./1lP:{1oe6D)FL#W2>X5L4&k3x{YgE41p+zi*ZUtg)ct5U1o[/8{4Wl((&NRc00AVi3krvl2MjRM<OnlE"
    "}$*o@@%n[S%m)rb}tGvB}40Vc<}wa69S=lz1{j=0{7Ece3J/k!]-k-)4ghhi]8w:[[ce\?j]-Cu%0Qo1+%m#Z30@I=%3Y5j1jMM1:Z90X7+m>rS@T]+RP#WMz"
    "903e[WDdx75B[#/7X^\?b5Xd@\?[1o{R&Tj3f2kEfR{6{:x(Lg#N{5KB^4\?fu[6C.wU<q3\?71%BG&}1S8t}3cP[g$%IS!DD<R{wBBLINaiq&]fxT6G4f8/xAa$"
    "Xch9}1qIZz^!a)D)!utB@16hC7#:3z9%4fc&vL#P<47M>[uhMN(olo:>4ITMjo>5c)-93F73W9[)eC]W7yFzq{#-ap7vp[+9pyP8@PB{q[!yVjZY1fj9oW[x"
    "1oeI+2pP:e(f[K0()sK$+o^Pw((XiF:ius23Cu\?INd{00{3@wt6RtHu]8FI6*:{t#S)}6A[VR^B4g1uY[bx%D1NuoUh.k)y}x%]lZP]T2ZPE(^+\?iEs+h\?:j"
    "59B6t4/Wyn[/ES4=Ok\?23JU7MjNhk6@Lv^{4!*La)G@^(@r069}T%HB3F$Tx3IaGk*T(F)2Oc}y\?Ul:s1pK#b1@W{N[D^+^{xpK101P2j]7pS71m^u83Jo2]"
    "@0-!K5+@nP1m=f]@jIm&(f53w(JC@D]wfYE{XK<[1S#y/1R\?%)7xa:teD(yN4IP(<&]/jm]Z<0{6z\?[n&sb.&@qc/!1ps-33jlN[@0iVe]hLzcjPkuV0r-r0"
    "1]XG6*5R9J}Vl^2JSogV}#.VG2(Leb{x7<b].ns.]-CTi3g4V$0@q%0%nArE@Siif]:yx{4/3211QB@c2ls.%1oXu2%nJf2@Qo{[3i7ni2Ngqt3Fk^:0r@o1"
    "@qDc/}2Qu\?@@Jre0Ruyc1o*B-3iQo^@R#Y#1PXP91ps&801n]b@qWAlFDM5N00iJh0q3i%3krJjjOP6A)gn^]@@h)0}2y<8><X-Y}uuY>}0XQ[}VdPH{Y79M"
    ")B.vk74\?1^5DQUQ58!c51o!Gj*6NiG1R{Li>Nmy^*WKXe{wBH9@TF@A4c[fb@qEe{%lY%=@0T8A5b^765b2C(6X9o!*U&.B=^8T$F+xxD}Wi62@{YE*0svG."
    "4g-px3f@n#81m}z1PFJ14iB{f-}f6C0rS922B\?a1joCoX4/s/U)/iyR[b-y!2Msx6}N3kh0r9eT2mqBt@@JO%8u/D{6-[EL4BK\?Q\?V<L]0N{Og%mbZx2hm&d"
    "\?t^MY]Y=hQ}U*VH{wyGY2>!e]1Q<}V2licP3&x5]1p>Uq@Srl9]8u60@pP**{vdU:@}2Zw{YYNJ1nR!A68Z!^}#3S(4fV5j5Zz3)2l0G*<O5YZ1oOe}2ljV5"
    "2xCR.joLoJ<]m*I8XjRf&0X-H9ng2/):g^.}t{3M5Bg+89LRQC0rScmP8n@@]dZpT5!U=p5EC@D}WqJZ5c-NC[B^}g5dm[z]8FYs(iePAYvSIK2/4XM6bp}%"
    ">OaG}(iE\?0}\?baE)E}d=}r[.f@pRDs@}#)>{x6pNdgdWg-Q(}j@0SP2Nf1=5Xj3Dj)!(}Y2hGd^@RuM$1mNoG[/}7xjK-*b2Mktq]zFjA3(epf5gmh:@VHRr"
    "4f#>h@#p){/-.6l4\?P3q0SzfB>lI1(FEgAD9l5<e4H8d!&q%X\?[acw-1nz3P5^!@ybh.Mb(/x5S03e/A[^:<=1m+4u4f>\?[{8s\?D{5<TU[xRss]EXC(aP.l4"
    "}1a<<ceFK%5C=4G4G[lN9sF]2g[T^IaoIrB3HFhh7v*5G)GK1r0-EZIjotl.8uFj&>)c\?s\?1QWL}$Qc$TM{0g@%no02(%X<1o3<t694\?68t+9[<o+RS1N#9\?"
    "5X[CT}tk:r)aA3n\?30=\?0(C9w1PXCX&SWEb5H4RT{.K[c3j=we5=y.<1Q>$p2[3.g{5b6![9G-M<})}e@PA{&]5]@(@r<bG1n@o02M1(h7Yol[]X!%j@@iVx"
    "\?TyWs0S@0c)d*Cw@i@[}jOx2h<l=Zx<]B#T1pLK#&0FD+@tkLPL>LB!2m6(y4*hd\?4fu!u0S9C/{4]l%5F^n3[@KgT@]a+r+>gSb]9+Tg3*EV\?(KAN)}VE34"
    "{V8qv1)t7c/9Q<A3JQeh]y&F5+(4Oe{6gZl@rk%J4H7h-3\?1b@7w[mn1k0\?&2GEei[-HR]}2yg584wf18tBH90QwT1}26%:+ixud{:YQtjOGuG[^lf60t2Lv"
    ")*P^55eLoO{5+iuPA25+F:AmV<)&AW5Bf8*<Q\?y<4G(Lc@@-L+5D&*Y5G7Gv}V3l]{Wwju}wx.H@{Hr^a(JR5{X1Q(2(a@u\?V9(V1{U:32lj+{ZwMHZ1o/bp"
    "{r&Qd5Ck\?@4e7k92N{!d)GhCQ}Xyvz@S&uA{5CZe{6.&k1O92@4FQ2j1n}&-67$8Z5sT31joC#u]cBm&0(1Ej0r9(a*x2$n}Vli/@o%<@Y+H.>0@Ia65c-/1"
    "4fV!g}2g{q}ucoP}V^Vm{Zmcl0$j&9%kTPt@0ik$<f/c/4^iPl)F=I$}uWVn0qV-V5^[{40r9]94*g$%0@-Ae{tL\?#@quDP2Lm^fgYvjZ]btBJ<](pX{Y6gE"
    "*Xg^K>(.B<}3NA14\?!Wo=>V=<0py0/1Eq[KjKrBa@P3c6^hqrdcL+UG7ZZDZ02b4)6Z(*>2LW@q\?q5z4!CwLW1{sq{4g0r]]+1[S@#\?jS1hja7.y<Qj&%Vk8"
    "@RCna}16Z/5zeb([5sU-6VZWy3K=>a5Dhtk7zL<b]8w:Q/Dges+JmEh6:hIteBZNZ*y<\?&8#py0/AC4c}Yjn6@Pa%!7C==.^sHxY%i}t}1Ud)Z{wTEZ9RD9g"
    "jP#M101Y826Am[i}$}q*1ps]g00h&/XbMNeF+xcJ}2yv5%l*6W}WSS90SzM>2kE]\?3Jy8e{Z3%8@Sr95}VlMJ@Tn#t[+9dX0TnD4%nr3d{X$f$&So2O@S92/"
    "1PnC#}tY:U2(s-c}#ViY4eHbe%mDl(%nI{9[=FQ{02(7r{XB))0r8##2)*Ly0rJt}2M>Oo00h#%2)fO}2cy!Vjo#04>l\?dY\?V@#R@osSu>jXY.0rg^H\?t^MI"
    "@}mox}uV%4@%X@b0osjV0062i3&wfx@pZQc<OP8^1O\?S)[Dey.@S@Yl6y=Q*4H@sm]z]44)d\?1!(JdF0(&mm+0riL]4Gum(1]!V1(\?7hO@}uM@{4!EB@qcP#"
    "}4&-i@S.u:]yqXM0Ru3^3&PX#@qu]@2]Jyv&rXPT1QLFi5%yz3jEb<I&Q]2e{yFOw3hau4}uW694c^-T2J(uB3>:PO4c!i8]:HTc*ZT[>5Fapv*wr901[Q{-"
    "1o4@U=JBWq2MLep(^2kk@rS9j{ve*&}]XN30}uwz0tjO*/yOCe10mQv(I}d<[+0mu1O\?=:0{*yo@<-aE@}#cX]7h\?$[adj)[=F<Q0p}oz110v6&Mhe()^f#-"
    "0Td&V[bA{i0Y]{FjsFLO+lFBJ01eYaq9TWO4!*q9<)08Z4h=eN4fC>!0#xX}f9roH8#.+h6C7rm)&BgPfz::X5-8ZB8tU=<*Vm9C3Ir{68wiy>[WoQ*5DR*w"
    "]-(8]2>5u^>[a#=2mH!f]xcRvbnH\?3@0I{a4I^.acoylC{SrDX6zSj1@RkrE5bK9V<PL/18u.o70P.Yj!D2-[3h-oW*vLQ\?9Gdcvj]LAV}UQE0@rh(<1]m/>"
    "@SAi80r@C.0r^#B0r@/6}#vGh]8!%*}2o4B00J92}#Wif}tpu]1p1q[1P/8e8U%jn@p{9/2M$b81n-610@Sn%@@%2r0rz])2KRJ<1h+co}UZH9{w&5r{YZ^4"
    "{v[:{@%eM50S@b#{5k9.1n}/Z3JQj%1oX8%1P!G90qVA/{4[^/0r&ub0@&xp}W3XbjL]/D6zh!l(<9.9@S&Vu/3h@r>kCbp]VN*o3NDwT+Op&W}4>dG1HFTK"
    "3(n:S{6zxe\?XV[b5EEye1@)sJ1Jf*6-R/OT{4)t*%igcf&n:wg43PuW0KRBz[9G/M2J(<j58!3c1nqAU]XnT10TW[rdcIc5}u=uY2F\?h(<np\?achcjP0Pb^W"
    "@p}\?@2dafWaQ!b}2JD0y6C}}T^{v^mjp}+<1k%vW@\?5#:@}DMQ!^\?#]].!W=3\?1C^]Z\?XW2mQwv}uWn:@{Y):5=8RB1Q+.n2mZ\?z(K0KX{.APr}owJ-0T/}u"
    "1Oh-Z2iD3H4*Jz(}${*5{WOZ!4h5*M{Yzoh%nA8]4AXOF}4J/v]-Tv.1OSb${vn4&}rFm*1Kc-l)^*w}@R(x{5Ft6i4/ctg2(>qv*y:8!1PE&%6iRPZj39/8"
    "<3vk&@PAwP&^RZ5\?[pq=3j!&P1O{[i)XP){3f]vEo$UJ>edxL]/60w+6AlXs&11or1tngi68R[c>i9iK!F6EW}XYXqjSL/W>O&<=5-:8$]e!qI}ow($fF%BN"
    ">Ltf31k--{i#xyf@}DQl+<k.JiRI(C.rg>1gdJFZ8^KnCf&XCfhX+-]}]e=*!GBw+([4(T4NG}a<YS<.jpqRy5:GeJ8}>2>{Yzuf*Zs]f[!gV:1o5F#Pz6c1"
    ">>>b<0%G7{5aow[3J/Ah6zz4<2mpUE78)IC]8Xd$)eN3S1{t\?J{v5E#81ds]3h:]t3G}V#FFeN*3igd+67ME\?*xNJu@{*f$>(]SC2NGt{0qt::c(8:!{yX2D"
    "}w5/3@}2)${xzih2]B0m{Yy%b3\?8pt2>)S<6z-4p2]aeAjpraY}$pM.bCa=d}ssZh)*Pj2]-$3*9Tz#LFFEO/]:gv82(b251P!Y01nJr61j^bv>LDe2+M4>J"
    "@%!=p{4m!C}Xfzr3J7p}0uR7r)ec.oObJY!0TO]71{LtN}%am12kF5(2&(0$]CN054Hq[n4Gl8i}1>c}]zOj\?@l>@F].wv>0R(/@4f3to68S482NpXb>Nu$:"
    "1p%&332ObtjOYo=2l+ef0rSDp3h$Ok0rSl9}vJVo@}3o@}T%N<75ejm3\?jlf@%5o5]7-4L1{>C1[+@s^}WS(j}${{a01fwl00rc81nzDY]A+E52kvV]2)6(e"
    "1RRdL]:*&{(&uIX@}(=}8#fmT6+7rU@qc9U&@PnW@@A051pbkd3K3h92Q5)B1aHyv}U/Q\?!c2Q@0$2hy@S9xg3kJ^N[zlz%jKKJl-}J^z{Y\?b=(iNv:eEp$r"
    "6y><46Vo\?C}TcGtaRUs.@tth6fx9c8)hDeNLl7%$3MF^u<O4Z96cWOT[=)>T5D82m5:*0Q@RMM:]zmnr0(&)%}}.}MXC2N/:O1T4{waZP2O4Bu1M]=(&qAJO"
    "2<&a000AI$1P!Yp\?$*7Y<luQI\?tNu^3p#6x^CbJD[+r.Q1O**Q^&O}#6bfR<jOX3%4gg$*>]h2F@{ZT94cWG#3Kld6}Ug0T0QYX1[:c5I)cBZ00$k8G3\?1Dj"
    "@WexZ3H)D^<jLg-aN/:h)gnp=):-3c\?V@GW*5\?DV1krX-}t7A]1NbW-}}.zR3Ih7n1{+tl0%f2f)HxBk2J#s5}s%:O/:)=Z<l}K8}sj7Fb]^(@(\?\?aW@16no"
    "0o<:(@qtst@Rc][2GTQ$5o^:dj.vxSk5tKckgC%vY}zjcj}[8vkp2l0ke)Rrka-+-j>ytuY=/2QZu>C!k27%ajN6z8ZmQysj+1YYj%e#<Y@<SUZ5-Z<kAzYN"
    "kd8joj$lpNjZYb(kiUM1Y/Dp7j{\?u@ki(Kdk4e^yY<qSxj#IJ9Y%fc7k2(i+j<m@3kgWekZk6a+j!<5[k1<3sZFA%BkKn9rj#8M{YB]opkw&-%k5WMQkkgDq"
    "j[9yvY*J^mY{3@xY(=S=Y=R{6ka(@gj%fr%j#kIyZt!14Z7LIpZk/ITZk8/-jT.\?Rj^&pZki\?%3kaNG#ke9&Qj(yg{YPCDwZfa*8ZknrojUNoGj!z%lZcU>H"
    "k87aKkbVNSZl<(!kzPrvjQNpYZfi-Oj$O$4kfAXsY<5%.k6my1Zgbu(k0QvmkaRP)ZlzFWj#})-j!=%gkuK=YZk8kFk06+&kl:cSkd@RCjf$yEZmGX8j)yRR"
    "k9pDwZrWByZcKmLZf>jEZTFOSZehcxZ7KFXj@].wkmO[!k6BGdkfnSTj%1w^Y:h^tjWG<ok6Z0EkcwLlZrB>ik1Uz8j)5JkjWJP.k4yX%ks>\?bZ4/FOZ2O$*"
    "Zw[3vkb&yckjrQ/k8CTWY(C$wjY+Glko5^WY!&PdkiAIDj}+M3j!k2:Z7vx=Y*vGXk70J8kw:sXkbEl:ZJuSvZeNsWk6U03Zj[/JZ2hfNZrCvmkHf%0kHK4@"
    "k7VcgZF}8+j)m\?-ZCpwBY+(8lkHp&tk8L^SY\?Y%Xkp6[Qj#VEDZNx[iZ5\?{^ka{6Yj<W{&k8C.fkucI:kG2G1j<j!hZ4!@PZYiJEjT$8Ejq64[8V/vz1]N*u"
    "*y1Oz5hrLf@r-e{7VtO](ez*I@2a}]]7&DL5A0sc.$<-V}3dkj(/!lA1mWz]2%4L)[=nB*ndBCO[9R!09oNkel&=L$1MY}9)\?tB83\?B#2[+Sw]3lxx[\?5#E@"
    "2K}K3F=0I03jvvu2OvdQ0ts#o[C}[l0u\?{M@}=M@)gYS@0sOb@mCDP3}#WMh)Cd{h]E1Lhi$yI/5A@Kx@se=g@TwX!f$/<$1N+KG}$yc$i1Dfr*wyHy<p:cA"
    "&^YX)5FjJi6aaZ266]&L2MBkC3LzXG2]h+f{tDR/9$dSh@tEr}]6OZa4DN=(0@SI@6Bs*l<sn4z7sq)Ud*)mg2}l5/(hO<03B{&H{UM$*6AZeW2MWN86^zPt"
    "{wCy1!^H{G\?xJv3/9}PRbK(0o5a*tg0%{k9jNh1{:P\?:Z0@S+XbmTw})9>}Z}vSY73MO%C[FRPy8UCP>@Rj(Z>&z)6bOPD41Ophw7-hO\?<P:x/]0nC=0@RV)"
    "JSGH<]bk)![GWq<FDUde2[^^46w&T13\?:fH].Gz<}qhF03F1[F}wZNSDhH[G[fEr62k)Yv)*x{44e/Yr@Oc=Bb(^/z8rz+33h2+HN&S<w83GTh2Qo-.3G]hJ"
    "joDiw/b<Ol2<h*74hFWF+nF3({3@fn%nr#m6Y5iw2N{Us5:v4A1I>*b<mSJ@(gT(/01!X@2mpLL}v&O]}04Fk5DRX43<W1q0RT5p(&d}j2kF!e@}L)>4!Q5e"
    "5Zzj%4IxAC&o<wA}S!w64K&Ff@}(yc4GU2W<{GMA}6C1#+<t&54KoZUfba(V0OXa[4^0@!]9=6k:o94q0^#Eqjpznu2HBL91{%L[{tk-zancmj5^mDI5E^hd"
    "YDlIH[:C=B[cxM#1ILt>aP&8800@Y24eZ553H[S259an}1pL+E5ap8w^^uR(}UihdbO5J16zi25@TY+z)*X:%&vafw7XV>[2LUK<5GQxw@l%GD0@Jc53Jrf*"
    "aNH9o@NI.&d\?qpy}p0B:-q{5U&uNWQ)/<ae2MaAh7weaW{UzU>jP+($@0SA8]zFT&0%FA3]-L^d@Te*T{YpZ24e71$0rJi20S<kn04m*H2MBY1@qD-(%nJf1"
    "0r@M7}V=]@2)HCc{5s<=0095%}VMQa{wLl)]-aQN0SA00@SAn%%mVJ@%nJ5$0Sixf}#cD}]:q9k@@&h]}t]%^}#MD(@sFS3{w+g3@S:On0TnY60rAPd{5Cf("
    "}tgo#00rx20g473j]Uuq@PsKV}v8SJ{WXE(755j+1]v(%@rZ*S3)2lN}Uo{:<]t\?/{q=[k0%n&00TeOT}$Z6^0Q*A\?{1Q5Q)!lXZ1]dS>\?2)9j@T6zkcH[It"
    "0}lmv{4!{}]-L71{6Zq:}tQf%@0h3B1)D9!]tpU71Oz#[1ooFz3g5b^{}EKD4d:G3}%%gK@ShP}03Qs})f&vX4f3d-aPc^1%ldQ/j.4I7Zk(A=jUI^#kd8<B"
    "ks$KjZK8zbZYjum"
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
