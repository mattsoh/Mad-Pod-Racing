// Search speed: runs the bot's think() on the states of a recorded game (replay.cpp's stdin) and reports
// evaluations per millisecond.
#define main bot_main
#include BOT_SOURCE
#undef main
#include <sstream>
using namespace mpr;
int main(int argc, char** argv) {
  double ms = argc > 1 ? std::atof(argv[1]) : 55;
  Track tr;
  std::cin >> tr.laps >> tr.n;
  for (int i = 0; i < tr.n; ++i) std::cin >> tr.x[i] >> tr.y[i];
  int grid;
  std::cin >> grid;
  State s;
  initState(s, tr, grid == 1);
  std::string line;
  std::getline(std::cin, line);
  Search S;
  S.tr = tr;
  S.P.oppShare = 0;
  NNPlayer nn(tr);
  S.model = [&nn](const State& st, int team, const Plan& other, Plan& p) { nn.rollout(st, team, other, p); };
  long evals = 0;
  double total = 0;
  for (int turn = 0; s.winner < 0 && turn < 120; ++turn) {
    double tx[4], ty[4];
    int th[4], kd[4];
    for (int k = 0; k < 4; ++k) {
      if (!std::getline(std::cin, line)) return 0;
      std::istringstream is(line);
      std::string w;
      is >> tx[k] >> ty[k] >> w;
      kd[k] = w == "BOOST" ? K_BOOST : w == "SHIELD" ? K_SHIELD : K_THRUST;
      th[k] = kd[k] == K_THRUST ? std::atoi(w.c_str()) : MAX_THRUST;
    }
    if (turn > 0) {
      State seen = s;
      for (int k = 0; k < 4; ++k) seen.p[k].angle = normAngle(roundHalfUp(s.p[k].angle)), seen.p[k].rad = seen.p[k].angle * DEG;
      Timer clk;
      S.think(seen, ms);
      total += clk.ms();
      evals += S.evals;
    }
    for (int k = 0; k < 4; ++k) applyCommand(s, k, tx[k], ty[k], th[k], kd[k], turn == 0);
    step(s, tr);
  }
  std::printf("%.1f evals/ms over %.0f ms\n", evals / total, total);
}
