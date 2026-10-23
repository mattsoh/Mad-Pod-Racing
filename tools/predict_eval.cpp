// How well can we predict a top bot's next move? Measured on real games.
//
// For positions sampled from games where one side is a top bot ("teacher"),
// each predictor guesses the teacher's two pods' commands; the guess is played
// through the engine (the other side plays its real commands) and compared with
// where the teacher's pods really ended up.
//
//   build/predict_eval --data data/games.txt --net nets/bc.bin --samples 1500 --search-ms 10
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/engine.h"
#include "../src/features.h"
#include "../src/search.h"
#include "nn_train.h"

using namespace mpr;

static bool g_hide = true;  // --hide 0: predict from the true state, as before the last-move inputs

struct Game {
  long id;
  Track tr;
  int rank[2];
  std::vector<std::vector<std::string>> cmds;
};

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

static std::vector<Game> readGames(const std::string& path) {
  std::vector<Game> games;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] != 'G') continue;
    std::istringstream h(line.substr(2));
    Game g;
    int w0, w1;
    h >> g.id >> g.tr.n;
    for (int i = 0; i < g.tr.n; ++i) h >> g.tr.x[i] >> g.tr.y[i];
    h >> g.rank[0] >> g.rank[1] >> w0 >> w1;
    g.tr.laps = 3;
    std::getline(in, line);
    int T = std::stoi(line);
    for (int k = 0; k < 4; ++k) std::getline(in, line);
    for (int t = 0; t < T; ++t) {
      std::vector<std::string> c(4);
      for (int k = 0; k < 4; ++k) std::getline(in, c[k]);
      g.cmds.push_back(c);
      for (int k = 0; k < 4; ++k) std::getline(in, line);
    }
    games.push_back(std::move(g));
  }
  return games;
}

// Puts team 1 in team 0's seat and vice versa.
static State swapTeams(const State& s) {
  State o = s;
  for (int k = 0; k < 4; ++k) o.p[k] = s.p[k ^ 2];
  std::swap(o.timeout[0], o.timeout[1]);
  std::swap(o.teamBoostUsed[0], o.teamBoostUsed[1]);
  if (o.winner >= 0) o.winner ^= 1;
  return o;
}

struct Score {
  long n = 0, rot = 0, kind = 0;
  double posErr = 0, velErr = 0;
};

int main(int argc, char** argv) {
  std::string data = "data/games.txt", netPath;
  int samples = 1500, teacherRank = 6, horizon = 1;
  bool allGames = false;
  double searchMs = 10;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto v = [&]() { return std::string(argv[++i]); };
    if (a == "--data") data = v();
    else if (a == "--net") netPath = v();
    else if (a == "--samples") samples = std::stoi(v());
    else if (a == "--search-ms") searchMs = std::stod(v());
    else if (a == "--teacher-rank") teacherRank = std::stoi(v());
    else if (a == "--horizon") horizon = std::stoi(v());
    else if (a == "--hide") g_hide = std::stoi(v()) != 0;
    else if (a == "--all") allGames = true;  // every game of --data (a held-out file), not bc's validation split
  }
  MLP net;
  bool haveNet = !netPath.empty() && loadNet(net, netPath);
  auto games = readGames(data);
  // validation games only (same split rule as tools/bc.cpp)
  std::vector<Game*> val;
  for (auto& g : games)
    if (allGames || (g.id * 2654435761u >> 7) % 1000 < 100) val.push_back(&g);
  printf("%zu validation games\n", val.size());

  const char* names[] = {"evolved (current bot)", "imitation network", "heuristic", "straight on (rot 0, 200)"};
  Score sc[4];
  Rng rng(5);
  int taken = 0;
  for (int it = 0; taken < samples && it < samples * 20; ++it) {
    Game& g = *val[rng.below((int)val.size())];
    int T = (int)g.cmds.size();
    if (T < 5) continue;
    if (T < horizon + 3) continue;
    int target = 1 + rng.below(T - 1 - horizon);
    int teacher = g.rank[0] <= teacherRank ? 0 : g.rank[1] <= teacherRank ? 1 : -1;
    if (teacher < 0) continue;
    // replay to the sampled turn
    State s, prev;
    initState(s, g.tr);
    for (int t = 0; t < target; ++t) {
      prev = s;
      for (int k = 0; k < 4; ++k) {
        double x, y;
        int th, kd;
        parseCmd(g.cmds[t][k], x, y, th, kd);
        applyCommand(s, k, x, y, th, kd, t == 0);
      }
      step(s, g.tr);
    }
    if (s.winner >= 0) continue;
    State real = s;
    // What our bot sees (it predicts from this): integer angles, last moves, no teacher shields
    for (int k = 0; k < 4; ++k) {
      for (State* x : {&s, &prev}) x->p[k].angle = normAngle(roundHalfUp(x->p[k].angle)), x->p[k].rad = x->p[k].angle * DEG;
      if (target > 1) observeLastMove(s.p[k], prev.p[k]);
      else clearLastMove(s.p[k]);
      if (k >> 1 == teacher && g_hide) s.p[k].shield = 0;
    }
    // the truth
    State truth = real;
    for (int k = 0; k < 4; ++k) {
      double x, y;
      int th, kd;
      parseCmd(g.cmds[target][k], x, y, th, kd);
      applyCommand(truth, k, x, y, th, kd, false);
    }
    State afterTruth = truth;
    step(afterTruth, g.tr);
    for (int d = 1; d < horizon; ++d) {
      for (int k = 0; k < 4; ++k) {
        double x, y;
        int th, kd;
        parseCmd(g.cmds[target + d][k], x, y, th, kd);
        applyCommand(afterTruth, k, x, y, th, kd, false);
      }
      step(afterTruth, g.tr);
    }
    Plan gaPlan;
    // predictions for the teacher's pods
    Gene pred[4][2];
    {
      Search S;
      S.tr = g.tr;
      S.rng = Rng(77 + taken);
      State me = teacher == 1 ? s : swapTeams(s);  // we sit in seat 0, the teacher in seat 1
      S.think(me, searchMs);
      pred[0][0] = S.theirs.g[0][0];
      pred[0][1] = S.theirs.g[0][1];
      gaPlan = S.theirs;
    }
    for (int j = 0; j < 2; ++j) {
      int k = 2 * teacher + j;
      if (haveNet) {
        float f[N_FULL], q[BC_OUT];
        fullFeatures(s, g.tr, k, f);
        net.forward(f, q);
        int br = 0, bk = 0;
        for (int i = 1; i < BC_ROT; ++i)
          if (q[i] > q[br]) br = i;
        for (int i = 1; i < BC_KIND; ++i)
          if (q[BC_ROT + i] > q[BC_ROT + bk]) bk = i;
        if (bk == 3 && !boostAvailable(s, k)) bk = 1;
        pred[1][j] = bcGene(br, bk);
      } else {
        pred[1][j] = {0, MAX_THRUST, K_THRUST};
      }
      int run, blk, orun, oblk;
      roles(s, g.tr, teacher, run, blk);
      roles(s, g.tr, 1 - teacher, orun, oblk);
      pred[2][j] = k == run ? runnerGene(s, g.tr, k) : blockerGene(s, g.tr, k, orun);
      pred[3][j] = {0, MAX_THRUST, K_THRUST};
    }
    auto predictMove = [&](int m, const State& cur, int k, int d) -> Gene {
      if (d == 0) return pred[m][k & 1];
      if (m == 0) return gaPlan.g[std::min(d, DEPTH - 1)][k & 1];
      if (m == 1 && haveNet) {
        float f[N_FULL], q[BC_OUT];
        fullFeatures(cur, g.tr, k, f);
        net.forward(f, q);
        int br = 0, bk = 0;
        for (int i = 1; i < BC_ROT; ++i)
          if (q[i] > q[br]) br = i;
        for (int i = 1; i < BC_KIND; ++i)
          if (q[BC_ROT + i] > q[BC_ROT + bk]) bk = i;
        if (bk == 3 && !boostAvailable(cur, k)) bk = 1;
        return bcGene(br, bk);
      }
      if (m == 2) {
        int run, blk, orun, oblk;
        roles(cur, g.tr, teacher, run, blk);
        roles(cur, g.tr, 1 - teacher, orun, oblk);
        return k == run ? runnerGene(cur, g.tr, k) : blockerGene(cur, g.tr, k, orun);
      }
      return Gene{0, MAX_THRUST, K_THRUST};
    };
    for (int m = 0; m < 4; ++m) {
      State p = s;
      for (int d = 0; d < horizon; ++d) {
        State cur = p;
        for (int k = 0; k < 4; ++k) {
          if ((k >> 1) == teacher) {
            applyGene(p, k, predictMove(m, cur, k, d));
          } else {
            double x, y;
            int th, kd;
            parseCmd(g.cmds[target + d][k], x, y, th, kd);
            applyCommand(p, k, x, y, th, kd, false);
          }
        }
        step(p, g.tr);
      }
      for (int j = 0; j < 2; ++j) {
        int k = 2 * teacher + j;
        double trueRot = angleDiff(s.p[k].angle, truth.p[k].angle);
        int tk;
        {
          double x, y;
          int th, kd;
          parseCmd(g.cmds[target][k], x, y, th, kd);
          if (kd == K_BOOST && !boostAvailable(s, k)) kd = K_THRUST, th = MAX_THRUST;
          tk = bcKind(th, kd);
        }
        sc[m].n++;
        sc[m].rot += bcRotBin(pred[m][j].rot) == bcRotBin(trueRot);
        sc[m].kind += bcKind(pred[m][j].thrust, pred[m][j].kind) == tk;
        sc[m].posErr += std::hypot(p.p[k].x - afterTruth.p[k].x, p.p[k].y - afterTruth.p[k].y);
        sc[m].velErr += std::hypot(p.p[k].vx - afterTruth.p[k].vx, p.p[k].vy - afterTruth.p[k].vy);
      }
    }
    ++taken;
  }
  printf("%d positions, %ld pod moves each, error measured %d turn(s) ahead\n", taken, sc[0].n, horizon);
  printf("%-28s %10s %10s %14s %14s\n", "predictor", "rotation", "thrust", "position err", "speed err");
  for (int m = 0; m < 4; ++m) {
    if (m == 1 && !haveNet) continue;
    printf("%-28s %9.1f%% %9.1f%% %12.1f u %12.1f u\n", names[m], 100.0 * sc[m].rot / sc[m].n,
           100.0 * sc[m].kind / sc[m].n, sc[m].posErr / sc[m].n, sc[m].velErr / sc[m].n);
  }
  return 0;
}
