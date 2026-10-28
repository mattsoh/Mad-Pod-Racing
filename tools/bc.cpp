// Imitation learning from top-bot replays.
//
// Every game in the dataset is re-simulated with the engine (any mismatch
// drops the game), and every move of a "teacher" pod (a bot ranked at or above
// --teacher-rank) becomes a sample: features from fullFeatures(), labels =
// rotation bin actually applied + thrust kind. Two softmax heads, trained with
// cross-entropy; samples are mirrored left-right to double the data.
//
//   build/bc --data data/games.txt --teacher-rank 6 --epochs 30 --out nets/bc
//   build/bc --data data/games.txt --eval nets/bc.bin
//   build/bc --data data/games.txt --init nets/bc160_t15.bin --epochs 12 --lr 3e-4 --out nets/bcl   # fine-tune
//
// Features are what our bot can see of the predicted pod's team at runtime: integer angles, no shields,
// last moves from observeLastMove (src/features.h). Without the shield state, a pod reloading after a
// SHIELD is labelled with the thrust it really had (0). --last-drop masks the last-move inputs of that
// share of the training samples, as rollouts do after their first step.
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/engine.h"
#include "../src/features.h"
#include "../src/rng.h"
#include "nn_train.h"

using namespace mpr;

struct Sample {
  float f[N_FULL];
  int8_t rot, kind;  // kind -1 = not a decision (pod shielded, thrust ignored)
  int game;
};

struct Game {
  long id;
  Track tr;
  int rank[2], won[2];
  std::vector<std::vector<std::string>> cmds;  // per turn, 4 commands
  std::vector<std::array<double, 24>> states;   // per turn+1, 4 x (x y vx vy ang next)
};

static bool readGames(const std::string& path, std::vector<Game>& games) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] != 'G') continue;
    std::istringstream h(line.substr(2));
    Game g;
    h >> g.id >> g.tr.n;
    for (int i = 0; i < g.tr.n; ++i) h >> g.tr.x[i] >> g.tr.y[i];
    h >> g.rank[0] >> g.rank[1] >> g.won[0] >> g.won[1];
    g.tr.laps = 3;
    int T;
    std::getline(in, line);
    T = std::stoi(line);
    auto readState = [&]() {
      std::array<double, 24> st;
      for (int k = 0; k < 4; ++k) {
        std::getline(in, line);
        std::istringstream s(line);
        for (int j = 0; j < 6; ++j) s >> st[k * 6 + j];
      }
      return st;
    };
    g.states.push_back(readState());
    for (int t = 0; t < T; ++t) {
      std::vector<std::string> c(4);
      for (int k = 0; k < 4; ++k) std::getline(in, c[k]);
      g.cmds.push_back(c);
      g.states.push_back(readState());
    }
    games.push_back(std::move(g));
  }
  return true;
}

static void parseCmd(const std::string& c, double& x, double& y, int& thrust, int& kind) {
  std::istringstream s(c);
  std::string tok;
  s >> x >> y >> tok;
  kind = K_THRUST;
  thrust = 0;
  if (tok == "SHIELD") kind = K_SHIELD;
  else if (tok == "BOOST") kind = K_BOOST;
  else thrust = std::atoi(tok.c_str());
}

static bool g_hideShields = true;

// Re-simulates a game; appends samples for teacher pods. Returns false on any mismatch.
static bool extract(const Game& g, int gi, int teacherRank, std::vector<Sample>& out, long& turnsOk) {
  State s;
  initState(s, g.tr);
  for (int k = 0; k < 4; ++k) {
    s.p[k].x = g.states[0][k * 6], s.p[k].y = g.states[0][k * 6 + 1];
  }
  bool teacher[2] = {g.rank[0] <= teacherRank, g.rank[1] <= teacherRank};
  std::vector<Sample> local;
  State prevSeen{};
  for (size_t t = 0; t < g.cmds.size(); ++t) {
    State before = s;
    State seen = s;  // as a bot sees it: integer angles and last moves
    for (int k = 0; k < 4; ++k) {
      seen.p[k].angle = normAngle(roundHalfUp(s.p[k].angle));
      seen.p[k].rad = seen.p[k].angle * DEG;
      if (t > 1) observeLastMove(seen.p[k], prevSeen.p[k]);
      else clearLastMove(seen.p[k]);
    }
    prevSeen = seen;
    State hidden[2] = {seen, seen};  // predicting team T: its shields are unknown to the other bot
    for (int T = 0; T < 2; ++T)
      for (int j = 2 * T; j < 2 * T + 2; ++j) hidden[T].p[j].shield = 0;
    for (int k = 0; k < 4; ++k) {
      double x, y;
      int thrust, kind;
      parseCmd(g.cmds[t][k], x, y, thrust, kind);
      double a0 = s.p[k].angle;
      bool boostOk = boostAvailable(s, k);
      int shieldBefore = s.p[k].shield;
      applyCommand(s, k, x, y, thrust, kind, t == 0);
      if (t == 0 || !teacher[k >> 1]) continue;
      Sample smp;
      fullFeatures(g_hideShields ? hidden[k >> 1] : before, g.tr, k, smp.f);
      if (!g_hideShields)
        for (int i = F_LAST; i < N_FULL; ++i) smp.f[i] = i == N_FULL - 1 ? 1.f : 0.f;  // old behaviour: no last move
      smp.rot = (int8_t)bcRotBin(angleDiff(a0, s.p[k].angle));
      if (kind == K_BOOST && !boostOk) kind = K_THRUST, thrust = MAX_THRUST;
      bool reloading = shieldBefore > 0 && kind != K_SHIELD;
      smp.kind = (int8_t)(reloading ? (g_hideShields ? bcKind(0, K_THRUST) : -1) : bcKind(thrust, kind));
      smp.game = gi;
      local.push_back(smp);
    }
    step(s, g.tr);
    const auto& nx = g.states[t + 1];
    for (int k = 0; k < 4; ++k) {
      const Pod& q = s.p[k];
      double da = std::fmod(nx[k * 6 + 4] - q.angle + 720.0, 360.0);
      if (q.x != nx[k * 6] || q.y != nx[k * 6 + 1] || q.vx != nx[k * 6 + 2] || q.vy != nx[k * 6 + 3] ||
          (da > 1e-3 && da < 360 - 1e-3) || q.next != (int)nx[k * 6 + 5])
        return false;
    }
    ++turnsOk;
  }
  out.insert(out.end(), local.begin(), local.end());
  return true;
}

static const int kMirrorY[] = {2, 4, 6, 8, 15, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37, 39, F_LAST};

static void mirror(const Sample& a, Sample& b) {
  b = a;
  for (int i : kMirrorY) b.f[i] = -a.f[i];
  b.rot = (int8_t)(BC_ROT - 1 - a.rot);
}

struct Acc {
  long n = 0, rot = 0, rotNear = 0, kn = 0, kind = 0;
  double loss = 0;
};

// Softmax cross-entropy over the two heads; fills dOut (scaled by 1/batch) and stats.
static void heads(const float* q, const Sample& s, float* dOut, float scale, Acc& acc) {
  auto softmax = [](const float* z, int n, float* p) {
    float m = z[0];
    for (int i = 1; i < n; ++i) m = std::max(m, z[i]);
    float sum = 0;
    for (int i = 0; i < n; ++i) sum += p[i] = std::exp(z[i] - m);
    for (int i = 0; i < n; ++i) p[i] /= sum;
  };
  float pr[BC_ROT], pk[BC_KIND];
  softmax(q, BC_ROT, pr);
  int br = 0;
  for (int i = 1; i < BC_ROT; ++i)
    if (pr[i] > pr[br]) br = i;
  acc.n++;
  acc.rot += br == s.rot;
  acc.rotNear += std::abs(br - s.rot) <= 1;
  acc.loss -= std::log(std::max(1e-9f, pr[(int)s.rot]));
  if (dOut)
    for (int i = 0; i < BC_ROT; ++i) dOut[i] = (pr[i] - (i == s.rot)) * scale;
  if (s.kind >= 0) {
    softmax(q + BC_ROT, BC_KIND, pk);
    int bk = 0;
    for (int i = 1; i < BC_KIND; ++i)
      if (pk[i] > pk[bk]) bk = i;
    acc.kn++;
    acc.kind += bk == s.kind;
    acc.loss -= std::log(std::max(1e-9f, pk[(int)s.kind]));
    if (dOut)
      for (int i = 0; i < BC_KIND; ++i) dOut[BC_ROT + i] = (pk[i] - (i == s.kind)) * scale;
  } else if (dOut) {
    for (int i = 0; i < BC_KIND; ++i) dOut[BC_ROT + i] = 0;
  }
}

static Acc evaluateSet(const MLP& net, const std::vector<Sample>& set) {
  Acc acc;
  float q[BC_OUT];
  for (auto& s : set) {
    net.forward(s.f, q);
    heads(q, s, nullptr, 0, acc);
  }
  return acc;
}

static void report(const char* name, const Acc& a) {
  printf("  %-6s samples %7ld  loss %.3f  rotation %.1f%%  thrust kind %.1f%%\n", name, a.n, a.loss / a.n,
         100.0 * a.rot / a.n, 100.0 * a.kind / std::max(1L, a.kn));
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string data = "data/games.txt", out = "nets/bc", evalFile, evalData;
  int teacherRank = 6, epochs = 30, hidden = 128, batch = 256, threads = 0;
  float lr = 1e-3f, valShare = 0.1f, lastDrop = 0.5f;
  std::string init;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto v = [&]() { return std::string(argv[++i]); };
    if (a == "--data") data = v();
    else if (a == "--out") out = v();
    else if (a == "--eval") evalFile = v();
    else if (a == "--eval-data") evalData = v();
    else if (a == "--teacher-rank") teacherRank = std::stoi(v());
    else if (a == "--epochs") epochs = std::stoi(v());
    else if (a == "--hidden") hidden = std::stoi(v());
    else if (a == "--batch") batch = std::stoi(v());
    else if (a == "--lr") lr = std::stof(v());
    else if (a == "--threads") threads = std::stoi(v());
    else if (a == "--init") init = v();
    else if (a == "--last-drop") lastDrop = std::stof(v());
    else if (a == "--hide-shields") g_hideShields = std::stoi(v()) != 0;
    else {
      printf("usage: bc --data FILE [--teacher-rank N] [--epochs N] [--hidden N] [--out PREFIX] [--eval FILE] [--eval-data FILE]"
             " [--init FILE] [--last-drop P] [--hide-shields 0|1]\n");
      return 2;
    }
  }
  std::vector<Game> games;
  if (!readGames(data, games)) return printf("cannot read %s\n", data.c_str()), 1;
  std::vector<Sample> train, val;
  long turnsOk = 0;
  int bad = 0;
  for (size_t gi = 0; gi < games.size(); ++gi) {
    std::vector<Sample> s;
    if (!extract(games[gi], (int)gi, teacherRank, s, turnsOk)) {
      if (bad < 10) printf("engine mismatch in game %ld\n", games[gi].id);
      ++bad;
      continue;
    }
    bool isVal = (games[gi].id * 2654435761u >> 7) % 1000 < valShare * 1000;  // split by game
    for (auto& x : s) {
      Sample m;
      mirror(x, m);
      (isVal ? val : train).push_back(x);
      if (!isVal) train.push_back(m);
    }
  }
  printf("games %zu (engine mismatch in %d), turns re-simulated exactly %ld\n", games.size(), bad, turnsOk);
  printf("teacher samples: train %zu (with mirrors), validation %zu\n", train.size(), val.size());
  {
    long hr[BC_ROT] = {}, hk[BC_KIND + 1] = {};
    for (auto& x : val) hr[(int)x.rot]++, hk[x.kind + 1]++;
    printf("validation rotation bins:");
    for (int i = 0; i < BC_ROT; ++i) printf(" %.1f%%", 100.0 * hr[i] / val.size());
    printf("\nvalidation thrust kinds (masked, 0, 200, SHIELD, BOOST):");
    for (int i = 0; i <= BC_KIND; ++i) printf(" %.1f%%", 100.0 * hk[i] / val.size());
    printf("\n");
  }

  std::vector<Sample> external;
  if (!evalData.empty()) {
    std::vector<Game> evalGames;
    if (!readGames(evalData, evalGames)) return printf("cannot read %s\n", evalData.c_str()), 1;
    long evalTurns = 0;
    for (size_t gi = 0; gi < evalGames.size(); ++gi) {
      std::vector<Sample> s;
      if (!extract(evalGames[gi], (int)gi, teacherRank, s, evalTurns)) continue;
      external.insert(external.end(), s.begin(), s.end());
    }
    printf("external evaluation: %zu samples from %zu games, %ld turns\n", external.size(), evalGames.size(), evalTurns);
  }

  if (!evalFile.empty()) {
    MLP net;
    if (!loadNet(net, evalFile)) return printf("cannot read %s\n", evalFile.c_str()), 1;
    report("val", evaluateSet(net, val));
    if (!external.empty()) report("external", evaluateSet(net, external));
    return 0;
  }

  int nth = threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency());
  Trainer T;
  T.net.init({N_FULL, hidden, hidden, BC_OUT}, 7);
  if (!init.empty()) {  // fine-tune; inputs the old network lacked start with zero weights
    MLP old;
    if (!loadNet(old, init)) return printf("cannot read %s\n", init.c_str()), 1;
    if (old.L.size() != T.net.L.size() || old.inputs() > N_FULL) return printf("%s: incompatible network\n", init.c_str()), 1;
    for (size_t l = 0; l < old.L.size(); ++l) {
      Dense& d = T.net.L[l];
      const Dense& o = old.L[l];
      if (o.out != d.out || (l > 0 && o.in != d.in)) return printf("%s: layer %zu differs\n", init.c_str(), l), 1;
      std::fill(d.W.begin(), d.W.end(), 0.f);
      for (int r = 0; r < d.out; ++r)
        for (int c = 0; c < o.in; ++c) d.W[(size_t)r * d.in + c] = o.W[(size_t)r * o.in + c];
      d.b = o.b;
    }
    printf("fine-tuning %s (%d inputs)\n", init.c_str(), old.inputs());
    report("val", evaluateSet(T.net, val));
  }
  T.setup(nth);
  Rng rng(99);
  std::vector<float> X((size_t)batch * N_FULL);
  std::vector<int> idx(train.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
  double best = 1e9;
  for (int ep = 1; ep <= epochs; ++ep) {
    for (size_t i = idx.size() - 1; i > 0; --i) std::swap(idx[i], idx[rng.below((int)i + 1)]);
    float lrNow = lr * (ep > epochs * 2 / 3 ? 0.2f : 1.f);
    std::vector<Acc> accs(nth);
    for (size_t b0 = 0; b0 + batch <= idx.size(); b0 += batch) {
      parallelFor(nth, batch, [&](int t, int lo, int hi) {
        int n = hi - lo;
        std::vector<float> xb((size_t)n * N_FULL), dOut((size_t)n * BC_OUT);
        for (int i = 0; i < n; ++i) {
          float* x = &xb[(size_t)i * N_FULL];
          std::memcpy(x, train[idx[b0 + lo + i]].f, sizeof(float) * N_FULL);
          uint32_t h = (uint32_t)(idx[b0 + lo + i] * 2654435761u) ^ (uint32_t)(ep * 40503u);
          if ((h >> 8) % 1000 < lastDrop * 1000)  // last move masked, as after a rollout's first step
            x[F_LAST] = x[F_LAST + 1] = x[F_LAST + 2] = 0.f, x[F_LAST + 3] = 1.f;
        }
        const float* q = T.workers[t].forward(T.net, xb.data(), n);
        for (int i = 0; i < n; ++i)
          heads(q + (size_t)i * BC_OUT, train[idx[b0 + lo + i]], &dOut[(size_t)i * BC_OUT], 1.f / batch, accs[t]);
        T.workers[t].backward(T.net, std::move(dOut), n);
      });
      T.adam(lrNow, 5.f);
    }
    Acc tr;
    for (auto& a : accs) tr.n += a.n, tr.rot += a.rot, tr.rotNear += a.rotNear, tr.kn += a.kn, tr.kind += a.kind, tr.loss += a.loss;
    Acc va = evaluateSet(T.net, val);
    printf("epoch %d\n", ep);
    report("train", tr);
    report("val", va);
    if (va.loss / va.n < best) {
      best = va.loss / va.n;
      saveNet(T.net, out + ".bin");
      exportHeader(T.net, out + "_weights.h", "imitation of top-" + std::to_string(teacherRank) + " bots, epoch " + std::to_string(ep));
    }
  }
  printf("best validation loss %.3f -> %s.bin\n", best, out.c_str());
  return 0;
}
