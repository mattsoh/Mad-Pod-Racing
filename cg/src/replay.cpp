// Rebuilds every turn of a game from its map and both players' commands, with the bot's own referee physics.
// stdin:  laps ncp, ncp lines "x y", grid (0 or 1), then per turn 4 lines "tx ty POWER|BOOST|SHIELD"
//         (player 0's two pods, then player 1's), until EOF
// stdout: per turn, before the commands are applied: "T turn to0 to1" then 4 lines
//         "x y vx vy angle next passed shield boostUsed", then "C c0 c1 c2 c3" (1 = that pod collided during the
//         turn), and finally "W winner turns"
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
  std::string line;
  std::getline(std::cin, line);
  for (int turn = 0; s.winner < 0; ++turn) {
    std::printf("T %d %d %d\n", turn, s.timeout[0], s.timeout[1]);
    for (int k = 0; k < 4; ++k) {
      const Pod& p = s.p[k];
      std::printf("%.0f %.0f %.0f %.0f %.4f %d %d %d %d\n", p.x, p.y, p.vx, p.vy, p.angle, p.next, p.passed, p.shield,
                  (int)p.boostUsed);
    }
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
    if (turn == 0)
      for (int k = 0; k < 4; ++k) {  // the referee points every pod at its first checkpoint before turn 0
        s.p[k].rad = std::atan2(tr.y[1 % tr.n] - s.p[k].y, tr.x[1 % tr.n] - s.p[k].x);
        s.p[k].angle = normAngle(s.p[k].rad / DEG);
      }
    for (int k = 0; k < 4; ++k) applyCommand(s, k, tx[k], ty[k], th[k], kd[k], turn == 0);
    State before = s;
    step(s, tr);
    std::printf("C");  // per pod: 1 if it touched another pod this turn (its move differs from moving alone)
    for (int k = 0; k < 4; ++k) {
      State iso = before;
      for (int j = 0; j < 4; ++j)
        if (j != k) iso.p[j].x = -1e6 * (j + 1), iso.p[j].y = -1e6, iso.p[j].vx = iso.p[j].vy = 0;
      iso.winner = -1;
      step(iso, tr);
      std::printf(" %d", iso.p[k].vx != s.p[k].vx || iso.p[k].vy != s.p[k].vy);
    }
    std::printf("\n");
  }
  std::printf("W %d %d\n", s.winner, s.turn);
}
