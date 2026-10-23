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

// Mad Pod Racing physics engine (Legend rules).
//
// Follows the Gold/Legend "Expert rules" and robostac's referee, which was
// checked against 500+ real CodinGame replays. Shared by the bot, the local
// arena and the trainer so all three see exactly the same game.
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

// The statement says one boost is shared by both pods, but real games give
// each pod its own (checked against 700+ CodinGame replays). Keep false.
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
// Signed difference b - a in [-180, 180).
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

// Resolves BOOST / SHIELD bookkeeping and returns the thrust that applies.
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

// Search-level move: rotation relative to the current heading.
inline void applyMove(State& s, int i, double rot, int thrust, int kind) {
  int t = resolveThrust(s, i, thrust, kind);
  Pod& p = s.p[i];
  rot = std::max(-MAX_ROT, std::min(MAX_ROT, rot));
  p.angle = normAngle(p.angle + rot);
  p.rad = p.angle * DEG;
  addThrust(p, t);
}

// Referee-level command: the pod turns toward (tx, ty).
inline void applyCommand(State& s, int i, double tx, double ty, int thrust, int kind, bool firstTurn) {
  int t = resolveThrust(s, i, thrust, kind);
  Pod& p = s.p[i];
  if (tx == p.x && ty == p.y) return;  // the referee skips rotation and thrust
  // Radians, unnormalised, exactly as the referee does it (matters at e.g. 270 degrees,
  // where cos() of the two representations rounds to opposite signs).
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

// Time in [0, ...) at which two pods touch; 0 if they already overlap; 10 if never.
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

// Elastic bounce with a minimum impulse of 120; a shielded pod weighs 10.
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

// Does the segment (x1,y1)-(x2,y2) pass within CP_RADIUS of (cx,cy)?
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

// Movement, collisions, checkpoints, friction, rounding, timeouts, result.
// Commands must already be applied with applyMove / applyCommand.
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

// Starting grid: pods side by side across the line from checkpoint 0 to 1.
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

// Race progress used by the search, the trainer and the referee's tie-break.
inline double progress(const Pod& q, const Track& tr) {
  double dx = tr.x[q.next] - q.x, dy = tr.y[q.next] - q.y;
  return q.passed * 50000.0 - std::sqrt(dx * dx + dy * dy);
}

}  // namespace mpr
// Plays the game with an embedded network. Two kinds are supported, told
// apart by their input/output sizes:
//
//  - imitation net (tools/bc.cpp): one pod's view of the whole race in, a
//    rotation (-18 / 0 / +18) and a thrust kind (0 / 200 / SHIELD / BOOST) out,
//    learned from the top Legend bots' moves;
//  - duel net (tools/train.cpp): Q-values for runner-vs-blocker action pairs,
//    solved as a matrix game each turn.
// Network inputs and actions for the runner-vs-blocker duel.
//
// Everything is expressed in the runner's frame: origin at the runner, +x
// pointing at its next checkpoint, so the same corner looks the same anywhere
// on the map.
#include <cmath>

// Evolutionary search over 6-turn plans for both pods.
//
// Each turn: (1) evolve a predicted plan for the opponent against our previous
// plan, (2) evolve our plan against that prediction. Plans are kept between
// turns, shifted by one, so the search keeps improving rather than restarting.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>

// Small, fast, seedable RNG shared by the bot, arena and trainer.
#include <cstdint>

namespace mpr {

// xorshift64*
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
    return s1 + P.robustW * std::min(0.0, s2 - s1);
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
// Zero-sum matrix game solver: rows maximise A, columns minimise it.
// Regret matching+ with alternating updates and linearly weighted averages.
#include <algorithm>

namespace mpr {

constexpr int GAME_MAX = 8;

// A is m x n, row-major. Writes mixed strategies x (rows) and y (columns) and
// returns the game value x^T A y.
inline float solveGame(const float* A, int m, int n, float* x, float* y, int iters = 150) {
  // A saddle point (pure equilibrium) is common and cheap to detect.
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
// Small dense network (leaky ReLU hidden layers, linear output), plus the
// int8 + Z85 text encoding used to carry its weights inside the source file.
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

  // Thread-safe: scratch space is per thread.
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

// ---- int8 quantisation ------------------------------------------------------
// Blob: "MPR1", u8 layers, (u16 in, u16 out) per layer, then per layer:
// per output row a float32 scale and `in` int8 weights, then float32 biases.

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

// ---- Z85 (ZeroMQ base85): no quotes or backslashes, safe in a C string ------

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
// No network embedded: the bot runs pure search.
// Replace with `build/train --export nets/duel.bin --to src/weights.h`.
inline const char* NN_WEIGHTS = "";

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

  void duel(const State& s, int run, int blk, float* xr, float* yb) const {
    float f[N_IN], q[N_OUT];
    duelFeatures(s, tr, run, blk, f);
    botNet().forward(f, q);
    solveGame(q, N_ACT, N_ACT, xr, yb);
  }

  // Moves for all four pods. Greedy when rng is null, sampled otherwise.
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
    // BOOST is not one of the duel net's actions: fire it on a long, aligned straight.
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
