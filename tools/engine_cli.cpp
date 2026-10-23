// Steps the engine on a scripted game, using the same stdin/stdout protocol as
// robostac's referee in `-test` mode, so the two can be diffed turn by turn.
//
// Input:  M, then M checkpoint lines (the track repeated 3 times plus the start),
//         then T, then per turn 4 ignored lines and 4 command lines "x y thrust".
// Output: per turn, 4 lines "x y vx vy angle next shield boosted".
#include <cstdio>
#include <cstring>
#include <string>

#include "../src/engine.h"

using namespace mpr;

int main(int argc, char** argv) {
  g_sharedBoost = argc > 1 && std::strcmp(argv[1], "--shared-boost") == 0;
  int m;
  if (scanf("%d", &m) != 1) return 1;
  double cx[64], cy[64];
  for (int i = 0; i < m; ++i) scanf("%lf %lf", &cx[i], &cy[i]);
  Track tr;
  tr.n = (m - 1) / 3;
  tr.laps = 3;
  for (int i = 0; i < tr.n; ++i) tr.x[i] = cx[i], tr.y[i] = cy[i];
  int turns;
  scanf("%d", &turns);
  State s;
  initState(s, tr);
  char buf[256];
  fgets(buf, sizeof buf, stdin);
  for (int t = 0; t < turns; ++t) {
    for (int k = 0; k < 4; ++k) fgets(buf, sizeof buf, stdin);  // ignored
    for (int k = 0; k < 4; ++k) {
      fgets(buf, sizeof buf, stdin);
      double x, y;
      char tok[32];
      sscanf(buf, "%lf %lf %31s", &x, &y, tok);
      int thrust = 0, kind = K_THRUST;
      if (!std::strcmp(tok, "SHIELD")) kind = K_SHIELD;
      else if (!std::strcmp(tok, "BOOST")) kind = K_BOOST;
      else thrust = std::stoi(tok);
      applyCommand(s, k, x, y, thrust, kind, t == 0);
    }
    step(s, tr);
    for (int k = 0; k < 4; ++k) {
      const Pod& q = s.p[k];
      int nextGlobal = q.passed >= tr.total() ? m - 1 : q.passed + 1;
      printf("%d %d %d %d %f %d %d %d\n", (int)q.x, (int)q.y, (int)q.vx, (int)q.vy, q.angle, nextGlobal,
             q.shield, q.boostUsed ? 1 : 0);
    }
  }
  return 0;
}
