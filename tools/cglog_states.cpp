// Rebuilds the per-turn states of a game from its commands, for battle logs from the CodinGame API (which
// hold both players' commands but no positions). Used by tools/cg_logs_to_dataset.py.
// Input:  N, N checkpoint lines "x y", T, then per turn 4 command lines "x y thrust|BOOST|SHIELD".
// Output: the 4 starting states, then per turn 4 states, as "x y vx vy angleDeg next" (data/games.txt).
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/engine.h"

using namespace mpr;

int main() {
  Track tr;
  tr.laps = 3;
  if (scanf("%d", &tr.n) != 1) return 1;
  for (int i = 0; i < tr.n; ++i)
    if (scanf("%lf %lf", &tr.x[i], &tr.y[i]) != 2) return 1;
  int T;
  if (scanf("%d", &T) != 1) return 1;
  State s;
  initState(s, tr);
  auto print = [&]() {
    for (int k = 0; k < 4; ++k)
      printf("%.0f %.0f %.0f %.0f %.9f %d\n", s.p[k].x, s.p[k].y, s.p[k].vx, s.p[k].vy, s.p[k].angle, s.p[k].next);
  };
  print();
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < 4; ++k) {
      double x, y;
      char w[32];
      if (scanf("%lf %lf %31s", &x, &y, w) != 3) return 1;
      int kind = !strcmp(w, "BOOST") ? K_BOOST : !strcmp(w, "SHIELD") ? K_SHIELD : K_THRUST;
      applyCommand(s, k, x, y, kind == K_THRUST ? atoi(w) : MAX_THRUST, kind, t == 0);
    }
    step(s, tr);
    print();
  }
  return 0;
}
