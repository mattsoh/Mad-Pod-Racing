// Nash-DQN trainer for the runner-vs-blocker duel (Agade & pb4's method).
//
// One network outputs Q[runner action][blocker action]. Each Bellman target
// solves the 7x7 zero-sum game at the next state: strategies from the online
// net, values from the target net ("double" Nash-DQN), with n-step returns.
// Runner reward: +1 per checkpoint, plus potential-based shaping toward the
// next checkpoint (which leaves the optimal policy unchanged); the blocker gets
// the negative. Running out of the 100-turn timeout ends the episode with a
// penalty.
//
//   build/train --blocker-prob 0   --steps 2000000 --out nets/runner      # stage 1: solo runner
//   build/train --init nets/runner.bin --steps 20000000 --out nets/duel   # stage 2: self-play duel
//   build/train --eval nets/duel.bin                                      # evaluate
//   build/train --export nets/duel.bin --to src/weights.h                 # embed in the bot
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/engine.h"
#include "../src/features.h"
#include "../src/maps.h"
#include "../src/nash.h"
#include "../src/nn.h"
#include "../src/search.h"
#include "nn_train.h"

using namespace mpr;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct Opt {
  long steps = 2000000;       // environment transitions
  int envs = 64;
  int batch = 256;
  int hidden = 128;
  float lr = 5e-4f;
  float gamma = 0.95f;
  float replayRatio = 4;      // samples drawn per transition
  int buffer = 1 << 19;
  int targetEvery = 250;      // gradient steps between target-net syncs
  int nstep = 3;              // n-step returns
  float shaping = 1.0f;       // weight of the distance-to-checkpoint potential
  float epsStart = 1.0f, epsEnd = 0.05f;
  long epsSteps = 300000;
  float blockerProb = 0.8f;   // share of episodes with a blocker
  float ambushProb = 0.4f;    // share of blocker episodes that start with the blocker waiting at the checkpoint
  float blockerEpsFloor = 0.15f;  // the blocker keeps exploring at least this much
  float demoProb = 0.3f;      // share of blocker episodes where the blocker follows the heuristic (demonstrations)
  float timeoutPenalty = 2.0f;
  int maxEpisode = 250;
  long evalEvery = 200000;
  int evalTracks = 40;
  uint64_t seed = 1;
  int threads = 0;            // 0 = all cores
  std::string init, out = "nets/duel", evalFile, exportFile, exportTo = "src/weights.h";
};

// ---------------------------------------------------------------------------
// Prioritised replay (proportional, sum tree)
// ---------------------------------------------------------------------------
struct Replay {
  int cap = 0, size = 0, pos = 0;
  std::vector<double> tree;
  std::vector<float> s, s2, r, disc;  // disc: discount applied to V(s2); 0 if terminal
  std::vector<uint8_t> ar, ab;
  double maxPrio = 1.0, alpha = 0.6;

  void init(int capacity) {
    cap = 1;
    while (cap < capacity) cap <<= 1;
    tree.assign(2 * (size_t)cap, 0.0);
    s.resize((size_t)cap * N_IN);
    s2.resize((size_t)cap * N_IN);
    r.resize(cap);
    ar.resize(cap);
    ab.resize(cap);
    disc.resize(cap);
  }
  void setPrio(int i, double p) {
    size_t k = (size_t)i + cap;
    double d = p - tree[k];
    for (; k; k >>= 1) tree[k] += d;
  }
  void push(const float* st, int a, int b, float rew, const float* st2, float d) {
    std::memcpy(&s[(size_t)pos * N_IN], st, N_IN * 4);
    std::memcpy(&s2[(size_t)pos * N_IN], st2, N_IN * 4);
    ar[pos] = (uint8_t)a;
    ab[pos] = (uint8_t)b;
    r[pos] = rew;
    disc[pos] = d;
    setPrio(pos, maxPrio);
    pos = (pos + 1) % cap;
    size = std::min(size + 1, cap);
  }
  int find(double v) const {
    size_t k = 1;
    while (k < (size_t)cap) {
      if (v <= tree[2 * k]) k = 2 * k;
      else v -= tree[2 * k], k = 2 * k + 1;
    }
    return (int)(k - cap);
  }
};

// ---------------------------------------------------------------------------
// Duel environment: runner = pod 0 (team 0), blocker = pod 2 (team 1).
// The track never ends; pods 1 and 3 are parked far away.
// ---------------------------------------------------------------------------
struct DuelEnv {
  Track tr;
  State s;
  bool hasBlk = false, demoBlk = false;
  int steps = 0;
  Rng rng;

  void park(int k) {
    Pod& p = s.p[k];
    p.x = -1e7 - k * 1e6;
    p.y = -1e7;
    p.vx = p.vy = 0;
  }

  void reset(float blockerProb, float ambushProb = 0, float demoProb = 0) {
    tr = anyTrack(rng);
    tr.laps = 100000;
    initState(s, tr);
    steps = 0;
    hasBlk = rng.uniform() < blockerProb;
    demoBlk = hasBlk && rng.uniform() < demoProb;
    int n = tr.n;
    if (rng.below(10) < 3) {
      // Real starting grid; first-turn rotation is free, so face checkpoint 1.
      int a = rng.below(2), b = 2 + rng.below(2);
      Pod r = s.p[a], bl = s.p[b];
      s.p[0] = r;
      s.p[2] = bl;
      for (int k : {0, 2}) s.p[k].angle = normAngle(std::atan2(tr.y[1 % n] - s.p[k].y, tr.x[1 % n] - s.p[k].x) / DEG);
    } else {
      int i = rng.below(n), pv = (i + n - 1) % n;
      double f = rng.uniform() * 0.9;
      Pod& R = s.p[0];
      R.x = std::round(tr.x[pv] + f * (tr.x[i] - tr.x[pv]) + rng.range(-1500, 1500));
      R.y = std::round(tr.y[pv] + f * (tr.y[i] - tr.y[pv]) + rng.range(-1500, 1500));
      double a0 = std::atan2(tr.y[i] - R.y, tr.x[i] - R.x) / DEG;
      R.angle = normAngle(a0 + rng.range(-120, 120));
      double sp = rng.range(0, 800), va = (a0 + rng.range(-70, 70)) * DEG;
      R.vx = std::trunc(sp * std::cos(va));
      R.vy = std::trunc(sp * std::sin(va));
      R.next = i;
      Pod& B = s.p[2];
      int where = rng.below(10);
      int c = where < 5 ? i : (i + 1) % n;
      if (where < 8) {
        B.x = std::round(tr.x[c] + rng.range(-2500, 2500));
        B.y = std::round(tr.y[c] + rng.range(-2500, 2500));
      } else {
        B.x = 1000 + rng.below(14001);
        B.y = 1000 + rng.below(7001);
      }
      if (std::hypot(B.x - R.x, B.y - R.y) < 820) B.x += 1700;
      double ba = rng.range(0, 360) * DEG, bs = rng.range(0, 500);
      B.vx = std::trunc(bs * std::cos(ba));
      B.vy = std::trunc(bs * std::sin(ba));
      B.angle = rng.range(0, 360);
      B.next = i;
    }
    if (hasBlk && rng.uniform() < ambushProb) {
      // Ambush: runner approaching its next checkpoint, blocker waiting on the way.
      int i = s.p[0].next;
      Pod& R = s.p[0];
      Pod& B = s.p[2];
      double a = rng.range(0, 360) * DEG, d = rng.range(3000, 7000);
      R.x = std::round(tr.x[i] + d * std::cos(a));
      R.y = std::round(tr.y[i] + d * std::sin(a));
      double toCp = std::atan2(tr.y[i] - R.y, tr.x[i] - R.x) / DEG;
      R.angle = normAngle(toCp + rng.range(-30, 30));
      double sp = rng.range(200, 700), va = (toCp + rng.range(-25, 25)) * DEG;
      R.vx = std::trunc(sp * std::cos(va));
      R.vy = std::trunc(sp * std::sin(va));
      double f = rng.range(0.55, 1.0);
      B.x = std::round(R.x + f * (tr.x[i] - R.x) + rng.range(-900, 900));
      B.y = std::round(R.y + f * (tr.y[i] - R.y) + rng.range(-900, 900));
      if (std::hypot(B.x - R.x, B.y - R.y) < 1200) B.x += (B.x > R.x ? 1 : -1) * 1200;
      B.angle = normAngle(std::atan2(R.y - B.y, R.x - B.x) / DEG + rng.range(-60, 60));
      double bs = rng.range(0, 250), ba = rng.range(0, 360) * DEG;
      B.vx = std::trunc(bs * std::cos(ba));
      B.vy = std::trunc(bs * std::sin(ba));
    }
    park(1);
    park(3);
    if (!hasBlk) park(2);
    s.timeout[0] = 40 + rng.below(61);
    s.timeout[1] = TIMEOUT_TURNS;
    bool used = rng.below(2);
    s.teamBoostUsed[0] = s.teamBoostUsed[1] = used;
  }

  void features(float* f) const { duelFeatures(s, tr, 0, hasBlk ? 2 : -1, f); }

  double potential() const {
    const Pod& R = s.p[0];
    return -std::hypot(tr.x[R.next] - R.x, tr.y[R.next] - R.y) / 10000.0;
  }

  // Returns the shaped reward; `raw` gets checkpoints passed this turn.
  float step(int ar, int ab, bool& terminal, bool& truncated, float& raw) {
    int before = s.p[0].passed;
    double phi0 = potential();
    applyGene(s, 0, actionGene(ar));
    if (hasBlk) applyGene(s, 2, actionGene(ab));
    step_(s, tr);
    s.timeout[1] = TIMEOUT_TURNS;  // the blocker's team never times out
    raw = (float)(s.p[0].passed - before);
    float rew = raw + (float)(shaping * (gamma * potential() - phi0));
    terminal = s.timeout[0] <= 0;
    if (terminal) rew -= penalty;
    truncated = ++steps >= maxEpisode;
    return rew;
  }
  static void step_(State& st, const Track& t) { mpr::step(st, t); }
  float penalty = 2, shaping = 1, gamma = 0.95f;
  int maxEpisode = 250;

  // n-step bookkeeping
  struct Pending {
    float s[N_IN];
    int a, b;
    float r;
  };
  std::vector<Pending> pend;
};

// Nearest discrete action to a heuristic move.
static int discreteAction(const Gene& g) {
  if (g.kind == K_SHIELD) return 6;
  int r = g.rot > 6 ? 2 : g.rot < -6 ? 0 : 1;
  return g.thrust >= 100 ? r : 3 + r;
}

static int argmaxOf(const float* p, int n) {
  int b = 0;
  for (int i = 1; i < n; ++i)
    if (p[i] > p[b]) b = i;
  return b;
}
static int sampleOf(const float* p, int n, Rng& r) {
  double u = r.uniform(), c = 0;
  for (int i = 0; i < n; ++i) {
    c += p[i];
    if (u < c) return i;
  }
  return argmaxOf(p, n);
}

// ---------------------------------------------------------------------------
// Evaluation on the real 3-lap race, from the starting grid
// ---------------------------------------------------------------------------
enum Policy { NET, HEUR };

struct EvalResult {
  double turns = 0;
  int finished = 0, timeouts = 0, games = 0;
};

// Runner (pod 0) races 3 laps; optional blocker (pod 2). Greedy play.
static EvalResult evalRace(const MLP* net, int nTracks, Policy runner, Policy blocker, bool withBlocker) {
  EvalResult res;
  Rng r(424242);
  for (int g = 0; g < nTracks; ++g) {
    Track tr = anyTrack(r);
    State s;
    initState(s, tr);
    for (int k : {1, 3}) s.p[k].x = -1e7 - k * 1e6, s.p[k].y = -1e7;
    if (!withBlocker) s.p[2].x = -2e7, s.p[2].y = -1e7;
    for (int k : {0, 2}) s.p[k].angle = normAngle(std::atan2(tr.y[1] - s.p[k].y, tr.x[1] - s.p[k].x) / DEG);
    int t = 0;
    for (; t < 600 && s.winner < 0; ++t) {
      float f[N_IN], q[N_OUT], x[N_ACT], y[N_ACT];
      Gene gr, gb{0, 0, K_THRUST};
      bool needNet = runner == NET || (withBlocker && blocker == NET);
      if (needNet) {
        duelFeatures(s, tr, 0, withBlocker ? 2 : -1, f);
        net->forward(f, q);
        solveGame(q, N_ACT, N_ACT, x, y);
      }
      gr = runner == NET ? actionGene(argmaxOf(x, N_ACT)) : runnerGene(s, tr, 0);
      if (withBlocker) gb = blocker == NET ? actionGene(argmaxOf(y, N_ACT)) : blockerGene(s, tr, 2, 0);
      applyGene(s, 0, gr);
      if (withBlocker) applyGene(s, 2, gb);
      step(s, tr);
      s.timeout[1] = TIMEOUT_TURNS;
      if (s.p[0].passed >= tr.total()) break;
    }
    res.games++;
    if (s.p[0].passed >= tr.total()) res.finished++, res.turns += t + 1;
    else if (s.timeout[0] <= 0) res.timeouts++, res.turns += 600;
    else res.turns += 600;
  }
  res.turns /= res.games;
  return res;
}

static void printEval(const MLP& net, int nTracks) {
  EvalResult solo = evalRace(&net, nTracks, NET, NET, false);
  EvalResult soloH = evalRace(nullptr, nTracks, HEUR, HEUR, false);
  EvalResult vsHeurBlk = evalRace(&net, nTracks, NET, HEUR, true);
  EvalResult heurVsNetBlk = evalRace(&net, nTracks, HEUR, NET, true);
  EvalResult self = evalRace(&net, nTracks, NET, NET, true);
  auto line = [](const char* name, const EvalResult& e) {
    printf("  %-34s avg turns %6.1f   finished %2d/%d   timeouts %2d\n", name, e.turns, e.finished, e.games,
           e.timeouts);
  };
  line("net runner, alone", solo);
  line("heuristic runner, alone", soloH);
  line("net runner vs heuristic blocker", vsHeurBlk);
  line("heuristic runner vs net blocker", heurVsNetBlk);
  line("net runner vs net blocker", self);
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------
static void train(const Opt& o) {
  Trainer T;
  if (!o.init.empty()) {
    if (!loadNet(T.net, o.init)) {
      fprintf(stderr, "cannot read %s\n", o.init.c_str());
      exit(1);
    }
    printf("loaded %s\n", o.init.c_str());
  } else {
    T.net.init({N_IN, o.hidden, o.hidden, N_OUT}, o.seed);
  }
  int threads = o.threads > 0 ? o.threads : (int)std::max(1u, std::thread::hardware_concurrency());
  T.setup(threads);
  MLP target = T.net;
  Replay R;
  R.init(o.buffer);
  std::vector<DuelEnv> envs(o.envs);
  for (int e = 0; e < o.envs; ++e) {
    envs[e].rng = Rng(o.seed * 7777 + e * 31 + 1);
    envs[e].penalty = o.timeoutPenalty;
    envs[e].maxEpisode = o.maxEpisode;
    envs[e].shaping = o.shaping;
    envs[e].gamma = o.gamma;
    envs[e].reset(o.blockerProb, o.ambushProb, o.demoProb);
  }
  Rng rng(o.seed * 99991 + 5);
  long steps = 0, gradSteps = 0, nextEval = o.evalEvery;
  double updDebt = 0, lossSum = 0, qSum = 0, cpSum = 0, toSum = 0;
  long lossN = 0, winSteps = 0;
  auto t0 = std::chrono::steady_clock::now();

  struct Transition {
    float s[N_IN], s2[N_IN];
    int a, b;
    float r, disc;
  };
  struct ActOut {
    std::vector<Transition> tr;
    double cp = 0, to = 0;
  };
  struct LearnOut {
    double loss = 0, q = 0;
  };
  std::vector<ActOut> actOut(threads);
  std::vector<LearnOut> learnOut(threads);
  std::vector<float> bs((size_t)o.batch * N_IN), bs2((size_t)o.batch * N_IN), q2o((size_t)o.batch * N_OUT),
      q2t((size_t)o.batch * N_OUT), tdv(o.batch), w(o.batch);
  std::vector<int> idx(o.batch);

  printf("training: %ld steps, %d envs, %d threads, net %d-%d-%d-%d, blocker share %.2f\n", o.steps, o.envs, threads,
         N_IN, o.hidden, o.hidden, N_OUT, o.blockerProb);
  while (steps < o.steps) {
    float eps = o.epsEnd + (o.epsStart - o.epsEnd) * std::max(0.0, 1.0 - (double)steps / o.epsSteps);

    // ---- act in every environment (each env has its own RNG) ----
    parallelFor(threads, o.envs, [&](int t, int lo, int hi) {
      ActOut& out = actOut[t];
      out.tr.clear();
      out.cp = out.to = 0;
      float f[N_IN], f2[N_IN], q[N_OUT], x[N_ACT], y[N_ACT];
      for (int e = lo; e < hi; ++e) {
        DuelEnv& E = envs[e];
        Rng& r = E.rng;
        E.features(f);
        T.net.forward(f, q);
        solveGame(q, N_ACT, N_ACT, x, y);
        int a = r.uniform() < eps ? r.below(N_ACT) : sampleOf(x, N_ACT, r);
        int b = r.uniform() < std::max(eps, o.blockerEpsFloor) ? r.below(N_ACT) : sampleOf(y, N_ACT, r);
        if (E.demoBlk && r.uniform() > 0.1) b = discreteAction(blockerGene(E.s, E.tr, 2, 0));
        bool term, trunc;
        float raw;
        float rew = E.step(a, b, term, trunc, raw);
        E.features(f2);
        DuelEnv::Pending pd;
        std::memcpy(pd.s, f, sizeof pd.s);
        pd.a = a, pd.b = b, pd.r = rew;
        E.pend.push_back(pd);
        // n-step transitions: emit when the window is full, flush at episode end.
        auto emit = [&](size_t i, bool terminalEnd) {
          Transition tt;
          float ret = 0, g = 1;
          for (size_t k = i; k < E.pend.size(); ++k) ret += g * E.pend[k].r, g *= o.gamma;
          std::memcpy(tt.s, E.pend[i].s, sizeof tt.s);
          std::memcpy(tt.s2, f2, sizeof tt.s2);
          tt.a = E.pend[i].a, tt.b = E.pend[i].b, tt.r = ret, tt.disc = terminalEnd ? 0.f : g;
          out.tr.push_back(tt);
        };
        if (term || trunc) {
          for (size_t i = 0; i < E.pend.size(); ++i) emit(i, term);
          E.pend.clear();
        } else if ((int)E.pend.size() == o.nstep) {
          emit(0, false);
          E.pend.erase(E.pend.begin());
        }
        out.cp += raw;
        out.to += term;
        if (term || trunc) E.reset(o.blockerProb, o.ambushProb, o.demoProb);
      }
    });
    for (auto& out : actOut) {
      for (auto& tt : out.tr) R.push(tt.s, tt.a, tt.b, tt.r, tt.s2, tt.disc);
      cpSum += out.cp;
      toSum += out.to;
    }
    steps += o.envs;
    winSteps += o.envs;

    // ---- learn ----
    updDebt += o.envs * o.replayRatio / o.batch;
    while (updDebt >= 1 && R.size >= 20000) {
      updDebt -= 1;
      double beta = std::min(1.0, 0.4 + 0.6 * (double)steps / o.steps);
      double total = R.tree[1], seg = total / o.batch, maxW = 0;
      for (int i = 0; i < o.batch; ++i) {
        int j = R.find(std::min(total * 0.999999, seg * (i + rng.uniform())));
        if (j >= R.size) j = rng.below(R.size);
        idx[i] = j;
        double pj = R.tree[(size_t)j + R.cap] / total;
        w[i] = (float)std::pow(R.size * pj, -beta);
        maxW = std::max(maxW, (double)w[i]);
        std::memcpy(&bs[(size_t)i * N_IN], &R.s[(size_t)j * N_IN], N_IN * 4);
        std::memcpy(&bs2[(size_t)i * N_IN], &R.s2[(size_t)j * N_IN], N_IN * 4);
      }
      parallelFor(threads, o.batch, [&](int t, int lo, int hi) {
        int n = hi - lo;
        forwardMany(T.net, &bs2[(size_t)lo * N_IN], n, &q2o[(size_t)lo * N_OUT]);
        forwardMany(target, &bs2[(size_t)lo * N_IN], n, &q2t[(size_t)lo * N_OUT]);
        const float* q = T.workers[t].forward(T.net, &bs[(size_t)lo * N_IN], n);
        std::vector<float> dOut((size_t)n * N_OUT, 0.f);
        LearnOut& lr = learnOut[t];
        for (int i = lo; i < hi; ++i) {
          int j = idx[i];
          float x[N_ACT], y[N_ACT], v = 0;
          if (R.disc[j] > 0) {
            solveGame(&q2o[(size_t)i * N_OUT], N_ACT, N_ACT, x, y);
            const float* qt = &q2t[(size_t)i * N_OUT];
            for (int a = 0; a < N_ACT; ++a)
              for (int b = 0; b < N_ACT; ++b) v += x[a] * y[b] * qt[a * N_ACT + b];
          }
          float targetQ = R.r[j] + R.disc[j] * v;
          int out = R.ar[j] * N_ACT + R.ab[j];
          float qv = q[(size_t)(i - lo) * N_OUT + out];
          float td = targetQ - qv;
          float wi = (float)(w[i] / maxW);
          float g = std::max(-1.f, std::min(1.f, td));  // Huber
          dOut[(size_t)(i - lo) * N_OUT + out] = -g * wi / o.batch;
          lr.loss += wi * (std::fabs(td) < 1 ? 0.5 * td * td : std::fabs(td) - 0.5);
          lr.q += qv;
          tdv[i] = td;
        }
        T.workers[t].backward(T.net, std::move(dOut), n);
      });
      for (int i = 0; i < o.batch; ++i) {
        double pr = std::pow(std::fabs(tdv[i]) + 0.01, R.alpha);
        R.setPrio(idx[i], pr);
        R.maxPrio = std::max(R.maxPrio, pr);
      }
      for (auto& lr : learnOut) lossSum += lr.loss, qSum += lr.q, lr = LearnOut();
      lossN += o.batch;
      T.adam(o.lr);
      if (++gradSteps % o.targetEvery == 0) target = T.net;
    }
    // ---- report ----
    if (steps >= nextEval || steps >= o.steps) {
      nextEval += o.evalEvery;
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      printf("\nsteps %ld  (%.0f/s, %.1f min)  grad steps %ld  eps %.3f\n", steps, steps / secs, secs / 60, gradSteps,
             eps);
      printf("  train: checkpoints per 100 turns %.2f  timeouts per 1000 turns %.2f  loss %.4f  mean Q %.3f\n",
             100 * cpSum / std::max(1L, winSteps), 1000 * toSum / std::max(1L, winSteps), lossSum / std::max(1L, lossN),
             qSum / std::max(1L, lossN));
      cpSum = toSum = lossSum = qSum = 0;
      lossN = winSteps = 0;
      printEval(T.net, o.evalTracks);
      saveNet(T.net, o.out + ".bin");
      saveNet(T.net, o.out + "_" + std::to_string(steps / 1000000) + "M.bin");  // keep every checkpoint
      exportHeader(T.net, o.out + "_weights.h", std::to_string(steps) + " steps");
    }
  }
  printf("\nsaved %s.bin and %s_weights.h\n", o.out.c_str(), o.out.c_str());
}

static void usage() {
  printf(
      "usage: train [options]\n"
      "  --steps N            environment transitions (default 2000000)\n"
      "  --envs N             parallel environments (64)\n"
      "  --hidden N           hidden width (128)\n"
      "  --batch N            batch size (256)\n"
      "  --lr X               Adam learning rate (5e-4)\n"
      "  --gamma X            discount (0.95)\n"
      "  --replay-ratio X     samples drawn per transition (4)\n"
      "  --buffer N           replay capacity (524288)\n"
      "  --target-every N     gradient steps between target syncs (250)\n"
      "  --nstep N            n-step returns (3)\n"
      "  --shaping X          weight of the distance-to-checkpoint potential (1)\n"
      "  --eps-steps N        steps over which exploration decays (300000)\n"
      "  --eps-start X / --eps-end X   exploration range (1.0 -> 0.05)\n"
      "  --blocker-prob X     share of episodes with a blocker (0.8; 0 = solo runner)\n"
      "  --ambush-prob X      share of those starting with the blocker waiting at the checkpoint (0.4)\n"
      "  --blocker-eps X      exploration floor for the blocker (0.15)\n"
      "  --demo-prob X        share of blocker episodes driven by the heuristic blocker (0.3)\n"
      "  --timeout-penalty X  reward when the runner times out (2)\n"
      "  --eval-every N       steps between evaluations (200000)\n"
      "  --init FILE          start from a saved .bin\n"
      "  --out PREFIX         writes PREFIX.bin and PREFIX_weights.h (nets/duel)\n"
      "  --seed N\n"
      "  --threads N          worker threads (default: all cores)\n"
      "  --eval FILE          evaluate a saved net and exit\n"
      "  --export FILE [--to src/weights.h]  write the bot's weight header and exit\n");
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // progress shows up in logs as it happens
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto v = [&]() -> std::string {
      if (i + 1 >= argc) usage(), exit(2);
      return argv[++i];
    };
    if (a == "--steps") o.steps = std::stol(v());
    else if (a == "--envs") o.envs = std::stoi(v());
    else if (a == "--hidden") o.hidden = std::stoi(v());
    else if (a == "--batch") o.batch = std::stoi(v());
    else if (a == "--lr") o.lr = std::stof(v());
    else if (a == "--gamma") o.gamma = std::stof(v());
    else if (a == "--replay-ratio") o.replayRatio = std::stof(v());
    else if (a == "--buffer") o.buffer = std::stoi(v());
    else if (a == "--target-every") o.targetEvery = std::stoi(v());
    else if (a == "--nstep") o.nstep = std::stoi(v());
    else if (a == "--shaping") o.shaping = std::stof(v());
    else if (a == "--eps-steps") o.epsSteps = std::stol(v());
    else if (a == "--eps-start") o.epsStart = std::stof(v());
    else if (a == "--eps-end") o.epsEnd = std::stof(v());
    else if (a == "--blocker-prob") o.blockerProb = std::stof(v());
    else if (a == "--ambush-prob") o.ambushProb = std::stof(v());
    else if (a == "--blocker-eps") o.blockerEpsFloor = std::stof(v());
    else if (a == "--demo-prob") o.demoProb = std::stof(v());
    else if (a == "--timeout-penalty") o.timeoutPenalty = std::stof(v());
    else if (a == "--eval-every") o.evalEvery = std::stol(v());
    else if (a == "--eval-tracks") o.evalTracks = std::stoi(v());
    else if (a == "--init") o.init = v();
    else if (a == "--out") o.out = v();
    else if (a == "--seed") o.seed = std::stoull(v());
    else if (a == "--threads") o.threads = std::stoi(v());
    else if (a == "--eval") o.evalFile = v();
    else if (a == "--export") o.exportFile = v();
    else if (a == "--to") o.exportTo = v();
    else return usage(), a == "-h" || a == "--help" ? 0 : 2;
  }
  if (!o.evalFile.empty() || !o.exportFile.empty()) {
    MLP net;
    std::string f = o.evalFile.empty() ? o.exportFile : o.evalFile;
    if (!loadNet(net, f)) return fprintf(stderr, "cannot read %s\n", f.c_str()), 1;
    if (!o.exportFile.empty()) {
      std::string q = quantize(net);
      MLP back;
      dequantize(q, back);
      exportHeader(net, o.exportTo, f);
      printf("wrote %s (%zu characters of weights)\nquantized net:\n", o.exportTo.c_str(), z85Encode(q).size());
      printEval(back, o.evalTracks);
    } else {
      printEval(net, o.evalTracks);
    }
    return 0;
  }
  std::string dir = o.out.substr(0, o.out.find_last_of('/'));
  if (dir != o.out) std::system(("mkdir -p " + dir).c_str());
  train(o);
  return 0;
}
