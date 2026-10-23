// Replays a real CodinGame game through the engine and reports any divergence.
// Input (stdin): N, N checkpoint lines, T, then per turn 4 state lines
// "x y vx vy angleDeg next" (state BEFORE the turn's commands) and 4 command lines.
// The final state line block (T+1-th) is the end state.
#include <cstdio>
#include <cstring>
#include <string>
#include "../src/engine.h"
using namespace mpr;
int main() {
  Track tr;
  scanf("%d", &tr.n);
  for (int i = 0; i < tr.n; ++i) scanf("%lf %lf", &tr.x[i], &tr.y[i]);
  tr.laps = 3;
  int T;
  scanf("%d", &T);
  State s;
  initState(s, tr);
  auto readState = [&](double v[4][6]) {
    for (int k = 0; k < 4; ++k) for (int j = 0; j < 6; ++j) scanf("%lf", &v[k][j]);
  };
  double cur[4][6];
  readState(cur);
  int bad = 0;
  for (int t = 0; t < T; ++t) {
    char line[256];
    int cmds = 0;
    for (int k = 0; k < 4; ++k) {
      double x, y;
      char tok[64];
      scanf("%lf %lf %63s", &x, &y, tok);
      fgets(line, sizeof line, stdin);
      int thrust = 0, kind = K_THRUST;
      if (!strcmp(tok, "SHIELD")) kind = K_SHIELD;
      else if (!strcmp(tok, "BOOST")) kind = K_BOOST;
      else thrust = atoi(tok);
      applyCommand(s, k, x, y, thrust, kind, t == 0);
      cmds++;
    }
    step(s, tr);
    double nx[4][6];
    readState(nx);
    for (int k = 0; k < 4; ++k) {
      const Pod& q = s.p[k];
      double a = fmod(nx[k][4] - q.angle + 720.0, 360.0);
      bool angOk = a < 1e-3 || a > 360 - 1e-3;
      if (q.x != nx[k][0] || q.y != nx[k][1] || q.vx != nx[k][2] || q.vy != nx[k][3] || !angOk || q.next != (int)nx[k][5]) {
        if (bad++ < 8)
          printf("turn %d pod %d: engine %.0f %.0f %.0f %.0f %.4f %d | game %.0f %.0f %.0f %.0f %.4f %d\n", t + 1, k, q.x, q.y,
                 q.vx, q.vy, q.angle, q.next, nx[k][0], nx[k][1], nx[k][2], nx[k][3], nx[k][4], (int)nx[k][5]);
        // resync to the real state so one error does not cascade
        Pod& w = s.p[k];
        w.x = nx[k][0], w.y = nx[k][1], w.vx = nx[k][2], w.vy = nx[k][3], w.angle = nx[k][4];
      }
    }
  }
  printf("turns %d, pod-turn mismatches %d\n", T, bad);
  return 0;
}
