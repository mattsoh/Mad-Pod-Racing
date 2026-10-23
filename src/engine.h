// Mad Pod Racing physics engine (Legend rules).
//
// Follows the Gold/Legend "Expert rules" and robostac's referee, which was
// checked against 500+ real CodinGame replays. Shared by the bot, the local
// arena and the trainer so all three see exactly the same game.
#pragma once
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
