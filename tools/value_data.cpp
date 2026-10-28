// Value-function dataset from real games: for every turn and side, the network's features seen from that
// side's runner (what our bot can see: own shields, enemy shields hidden, last moves) and how the race lead
// changed over the next H turns. The search can then judge a whole 4-pod situation, not only role terms.
//
//   build/value_data --data data/games.txt --out data/value.bin [--horizon 30]
// Output: float32 records of N_FULL features, target (lead change in checkpoints, clipped to +-3; +-3 when
// the side wins or loses within the horizon), then int32 game id and side.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/engine.h"
#include "../src/features.h"

using namespace mpr;

int main(int argc, char** argv) {
  std::string data = "data/games.txt", out = "data/value.bin";
  int H = 30;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--data") data = argv[++i];
    else if (a == "--out") out = argv[++i];
    else if (a == "--horizon") H = std::atoi(argv[++i]);
  }
  std::ifstream in(data);
  FILE* fo = std::fopen(out.c_str(), "wb");
  if (!in || !fo) return std::printf("cannot open files\n"), 1;
  std::string line;
  long rows = 0, games = 0, bad = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] != 'G') continue;
    std::istringstream h(line.substr(2));
    long id;
    Track tr;
    int rk[2], won[2];
    h >> id >> tr.n;
    for (int i = 0; i < tr.n; ++i) h >> tr.x[i] >> tr.y[i];
    h >> rk[0] >> rk[1] >> won[0] >> won[1];
    tr.laps = 3;
    std::getline(in, line);
    int T = std::stoi(line);
    std::vector<std::array<std::string, 4>> cmds(T);
    for (int k = 0; k < 4; ++k) std::getline(in, line);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < 4; ++k) std::getline(in, cmds[t][k]);
      for (int k = 0; k < 4; ++k) std::getline(in, line);
    }
    // re-simulate, keeping each turn's state
    std::vector<State> st;
    State s;
    initState(s, tr);
    for (int t = 0; t < T && s.winner < 0; ++t) {
      st.push_back(s);
      for (int k = 0; k < 4; ++k) {
        std::istringstream c(cmds[t][k]);
        double x, y;
        std::string w;
        c >> x >> y >> w;
        int kind = w == "BOOST" ? K_BOOST : w == "SHIELD" ? K_SHIELD : K_THRUST;
        applyCommand(s, k, x, y, kind == K_THRUST ? std::atoi(w.c_str()) : MAX_THRUST, kind, t == 0);
      }
      step(s, tr);
    }
    st.push_back(s);
    int n = (int)st.size();
    if (n < 10) { ++bad; continue; }
    int winner = won[0] ? 0 : won[1] ? 1 : -1;
    auto lead = [&](const State& x, int side) {
      double a = std::max(progress(x.p[2 * side], tr), progress(x.p[2 * side + 1], tr));
      double b = std::max(progress(x.p[2 - 2 * side], tr), progress(x.p[3 - 2 * side], tr));
      return (a - b) / 50000.0;
    };
    for (int t = 2; t + 1 < n; ++t) {
      for (int side = 0; side < 2; ++side) {
        State v = st[t];
        for (int k = 0; k < 4; ++k) {
          v.p[k].angle = normAngle(roundHalfUp(st[t].p[k].angle));
          v.p[k].rad = v.p[k].angle * DEG;
          Pod prev = st[t - 1].p[k];
          prev.angle = normAngle(roundHalfUp(prev.angle));
          observeLastMove(v.p[k], prev);
          if (k >> 1 != side) v.p[k].shield = 0;
        }
        int run = progress(v.p[2 * side], tr) >= progress(v.p[2 * side + 1], tr) ? 2 * side : 2 * side + 1;
        float rec[N_FULL + 1];
        fullFeatures(v, tr, run, rec);
        double target;
        if (t + H < n) target = lead(st[t + H], side) - lead(st[t], side);
        else if (winner >= 0) target = winner == side ? 3 : -3;
        else target = lead(st[n - 1], side) - lead(st[t], side);
        rec[N_FULL] = (float)std::max(-3.0, std::min(3.0, target));
        int32_t meta[2] = {(int32_t)(id % 2000000000), side};
        std::fwrite(rec, sizeof(float), N_FULL + 1, fo);
        std::fwrite(meta, sizeof(int32_t), 2, fo);
        ++rows;
      }
    }
    ++games;
  }
  std::fclose(fo);
  std::printf("%ld games (%ld skipped), %ld rows (%d floats + 2 ints each) -> %s\n", games, bad, rows, N_FULL + 1, out.c_str());
  return 0;
}
