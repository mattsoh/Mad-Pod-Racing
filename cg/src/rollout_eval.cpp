// How well the bot predicts the opponent over its search horizon. For a game (replay.cpp's stdin, then a last
// line "me SEAT"), from every 2nd turn: roll the opponent's pods forward DEPTH turns with the predictor while our
// pods replay their recorded commands, and compare with what really happened.
// stdout per (turn, depth): "turn depth errNN errHeur" = mean distance of the opponent's 2 pods from where they were.
#define main bot_main
#include BOT_SOURCE
#undef main
#include <sstream>
#include <vector>
using namespace mpr;
struct Cmd { double tx, ty; int th, kd; };
int main() {
  Track tr;
  std::cin >> tr.laps >> tr.n;
  for (int i = 0; i < tr.n; ++i) std::cin >> tr.x[i] >> tr.y[i];
  int grid;
  std::cin >> grid;
  std::string line;
  std::getline(std::cin, line);
  std::vector<std::array<Cmd, 4>> cmds;
  int me = 0;
  while (std::getline(std::cin, line)) {
    if (line.rfind("me ", 0) == 0) { me = std::atoi(line.c_str() + 3); break; }
    std::array<Cmd, 4> c;
    for (int k = 0; k < 4; ++k) {
      if (k) std::getline(std::cin, line);
      std::istringstream is(line);
      std::string w;
      is >> c[k].tx >> c[k].ty >> w;
      c[k].kd = w == "BOOST" ? K_BOOST : w == "SHIELD" ? K_SHIELD : K_THRUST;
      c[k].th = c[k].kd == K_THRUST ? std::atoi(w.c_str()) : MAX_THRUST;
    }
    cmds.push_back(c);
  }
  std::vector<State> st;
  State s;
  initState(s, tr, grid == 1);
  for (size_t t = 0; t < cmds.size() && s.winner < 0; ++t) {
    st.push_back(s);
    for (int k = 0; k < 4; ++k) applyCommand(s, k, cmds[t][k].tx, cmds[t][k].ty, cmds[t][k].th, cmds[t][k].kd, t == 0);
    step(s, tr);
  }
  st.push_back(s);
  NNPlayer nn(tr);
  int op = 1 - me;
  for (size_t t = 1; t + DEPTH < st.size(); t += 2) {
    for (int mode = 0; mode < 2; ++mode) {
      State x = st[t];
      for (int k = 0; k < 4; ++k) {  // what our bot knows: integer angles, no enemy shield state
        x.p[k].angle = normAngle(roundHalfUp(x.p[k].angle));
        x.p[k].rad = x.p[k].angle * DEG;
        if (k >> 1 == op) x.p[k].shield = 0;
#ifdef MPR_LAST_MOVE
        Pod& p = x.p[k];
        const Pod& q = st[t - 1].p[k];
        double qa = normAngle(roundHalfUp(q.angle));
        p.lastRot = p.lastThrust = 0, p.lastTouch = p.lastMasked = false;
        if (t > 1) {
          double c = std::cos(p.angle * DEG), sn = std::sin(p.angle * DEG);
          double ax = p.vx / FRICTION - q.vx, ay = p.vy / FRICTION - q.vy;
          double along = ax * c + ay * sn, side = -ax * sn + ay * c;
          p.lastRot = (float)angleDiff(qa, p.angle);
          p.lastTouch = std::fabs(side) > 2.5 + 0.01 * std::fabs(along) || along < -3 || (along > 203 && std::fabs(along - 650) > 3);
          p.lastThrust = p.lastTouch ? 0.f : (float)along;
        }
#endif
      }
      for (int d = 0; d < DEPTH; ++d) {
        Gene g[2];
        if (mode == 0) {
          for (int j = 0; j < 2; ++j) g[j] = nn.bcMove(x, 2 * op + j, nullptr);
        } else {
          Plan h = heuristicPlan(x, tr, op);
          g[0] = h.g[0][0], g[1] = h.g[0][1];
        }
        for (int k = 0; k < 4; ++k) {
          if (k >> 1 == op) applyGene(x, k, g[k & 1]);
          else applyCommand(x, k, cmds[t + d][k].tx, cmds[t + d][k].ty, cmds[t + d][k].th, cmds[t + d][k].kd, false);
        }
        step(x, tr);
        const State& real = st[t + d + 1];
        double e = 0;
        for (int j = 0; j < 2; ++j) e += 0.5 * dist(x.p[2 * op + j].x, x.p[2 * op + j].y, real.p[2 * op + j].x, real.p[2 * op + j].y);
        std::printf("%zu %d %d %.1f\n", t, d + 1, mode, e);
        if (x.winner >= 0) break;
      }
    }
  }
}
