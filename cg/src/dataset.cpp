// Per turn and pod of a game: the network's input features, the move the pod made, and what the embedded network
// and the heuristic predicted. Same stdin as replay.cpp. stdout, one line per pod-turn (turn 0 skipped, its
// rotation is free):
//   turn pod role rot thrust kind | 44 features | 7 network probabilities | heuristic rot thrust kind
//   | last turn's move as seen from outside: rotation (from the integer angles), thrust (from the speed change, -1
//     when the pod touched another pod), touched (0/1)
// role: 0 runner, 1 blocker. rot in degrees (-18..18), kind 0 thrust 1 shield 2 boost.
#define main bot_main
#include BOT_SOURCE
#undef main
#include <sstream>
using namespace mpr;
int main() {
  Track tr;
  std::cin >> tr.laps >> tr.n;
  for (int i = 0; i < tr.n; ++i) std::cin >> tr.x[i] >> tr.y[i];
  int grid;
  std::cin >> grid;
  State s;
  initState(s, tr, grid == 1);
  NNPlayer nn(tr);
  std::string line;
  std::getline(std::cin, line);
  State prevSeen{};
  bool prevHit[4] = {};
  int me = std::getenv("ME") ? std::atoi(std::getenv("ME")) : -1;  // our seat: the enemy's shields are hidden from us
  for (int turn = 0; s.winner < 0; ++turn) {
    double tx[4], ty[4];
    int th[4], kd[4];
    bool ok = true;
    for (int k = 0; k < 4 && ok; ++k) {
      if (!std::getline(std::cin, line)) { ok = false; break; }
      std::istringstream is(line);
      std::string w;
      is >> tx[k] >> ty[k] >> w;
      kd[k] = w == "BOOST" ? K_BOOST : w == "SHIELD" ? K_SHIELD : K_THRUST;
      th[k] = kd[k] == K_THRUST ? std::atoi(w.c_str()) : MAX_THRUST;
    }
    if (!ok) break;
    if (turn > 0) {
      State seen = s;  // what the bots see: integer angles
      for (int k = 0; k < 4; ++k) {
        seen.p[k].angle = normAngle(roundHalfUp(s.p[k].angle)), seen.p[k].rad = seen.p[k].angle * DEG;
        if (me >= 0 && k >> 1 != me) seen.p[k].shield = 0;
      }
      Plan h[2] = {heuristicPlan(seen, tr, 0), heuristicPlan(seen, tr, 1)};
      for (int k = 0; k < 4; ++k) {
        const Pod& p = seen.p[k];
        int run, blk;
        roles(seen, tr, k >> 1, run, blk);
        double want = std::atan2(ty[k] - s.p[k].y, tx[k] - s.p[k].x) / DEG;
        double rot = std::max(-MAX_ROT, std::min(MAX_ROT, angleDiff(s.p[k].angle, want)));
        bool still = tx[k] == s.p[k].x && ty[k] == s.p[k].y;
        if (still) rot = 0;
        int kind = kd[k] == K_SHIELD ? 1 : kd[k] == K_BOOST ? 2 : 0;
        int thrust = kd[k] == K_SHIELD ? 0 : kd[k] == K_BOOST ? (boostAvailable(seen, k) ? 650 : 200) : th[k];
        if (s.p[k].shield > 0 && kind == 0) thrust = 0;  // engine reloading after a shield
        if (s.p[k].shield > 0 && kind == 0) kind = 3;    // ...flagged so training can tell (kind 3 = reloading)
        float f[N_FULL], pr[BC_OUT];
        fullFeatures(seen, tr, k, f);
        nn.bcMove(seen, k, nullptr, pr);
        const Gene& g = h[k >> 1].g[0][k & 1];
        std::printf("%d %d %d %.3f %d %d |", turn, k, k == run ? 0 : 1, rot, thrust, kind);
        for (int i = 0; i < N_FULL; ++i) std::printf(" %.5g", f[i]);
        std::printf(" |");
        for (int i = 0; i < BC_OUT; ++i) std::printf(" %.4f", pr[i]);
        std::printf(" | %.2f %d %d", g.rot, g.thrust, g.kind == K_SHIELD ? 1 : g.kind == K_BOOST ? 2 : 0);
        const Pod& q = prevSeen.p[k];
        double lastRot = turn > 1 ? angleDiff(q.angle, p.angle) : 0;
        double c = std::cos(p.angle * DEG), sn = std::sin(p.angle * DEG);
        // without a touch the speed change points along the heading (to within the truncation of the speeds),
        // which the bot can check at runtime too: a sideways residual or an impossible thrust means a touch
        double ax = p.vx / FRICTION - q.vx, ay = p.vy / FRICTION - q.vy;
        double along = ax * c + ay * sn, side = -ax * sn + ay * c;
        bool seenTouch = turn > 1 && (std::fabs(side) > 2.5 + 0.01 * std::fabs(along) || along < -3 || (along > 203 && std::fabs(along - 650) > 3));
        double lastThrust = turn > 1 && !seenTouch ? along : -1;
        std::printf(" | %.0f %.1f %d %d\n", lastRot, lastThrust, (int)seenTouch, (int)prevHit[k]);
      }
    }
    for (int k = 0; k < 4; ++k) {
      prevSeen.p[k] = s.p[k];
      prevSeen.p[k].angle = normAngle(roundHalfUp(s.p[k].angle));
    }
    for (int k = 0; k < 4; ++k) applyCommand(s, k, tx[k], ty[k], th[k], kd[k], turn == 0);
    State before = s;
    step(s, tr);
    for (int k = 0; k < 4; ++k) {  // touched another pod: its move differs from moving alone
      State iso = before;
      for (int j = 0; j < 4; ++j)
        if (j != k) iso.p[j].x = -1e6 * (j + 1), iso.p[j].y = -1e6, iso.p[j].vx = iso.p[j].vy = 0;
      step(iso, tr);
      prevHit[k] = iso.p[k].vx != s.p[k].vx || iso.p[k].vy != s.p[k].vy;
    }
  }
}
