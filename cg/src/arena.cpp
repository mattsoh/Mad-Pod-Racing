// Plays two bot executables against each other with the bot's referee physics, several games in parallel.
//   arena BOT_A BOT_B GAMES [THREADS] [MAPS_FILE] [SEED]
// Maps file: one map per line, "x1 y1 x2 y2 ..." (taken from real battles). Each map is played twice, once from
// each side of the grid. Prints one line per game and the final score of A. Bots inherit the environment
// (MPR_TIME_MS etc.); ENV_A / ENV_B ("K=V,K=V") add variables for one side only.
#define main bot_main
#include BOT_SOURCE
#undef main
#include <atomic>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace mpr;

struct Proc {
  pid_t pid = -1;
  int in = -1, out = -1;
  std::string buf;
  bool start(const std::string& exe, const std::string& env) {
    int a[2], b[2];
    if (pipe(a) || pipe(b)) return false;
    pid = fork();
    if (pid == 0) {
      dup2(a[0], 0);
      dup2(b[1], 1);
      int devnull = open("/dev/null", O_WRONLY);
      dup2(devnull, 2);
      close(a[1]);
      close(b[0]);
      std::stringstream ss(env);
      std::string kv;
      while (std::getline(ss, kv, ','))
        if (auto e = kv.find('='); e != std::string::npos) setenv(kv.substr(0, e).c_str(), kv.substr(e + 1).c_str(), 1);
      execl(exe.c_str(), exe.c_str(), (char*)nullptr);
      _exit(127);
    }
    close(a[0]);
    close(b[1]);
    in = a[1];
    out = b[0];
    return true;
  }
  void send(const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
      ssize_t w = write(in, s.data() + off, s.size() - off);
      if (w <= 0) return;
      off += w;
    }
  }
  bool line(std::string& l, int ms) {
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      auto nl = buf.find('\n');
      if (nl != std::string::npos) {
        l = buf.substr(0, nl);
        buf.erase(0, nl + 1);
        return true;
      }
      int left = ms - (int)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (left <= 0) return false;
      pollfd p{out, POLLIN, 0};
      if (poll(&p, 1, left) <= 0) return false;
      char c[4096];
      ssize_t r = read(out, c, sizeof c);
      if (r <= 0) return false;
      buf.append(c, r);
    }
  }
  void stop() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
    if (in >= 0) close(in);
    if (out >= 0) close(out);
    pid = -1;
  }
};

struct Result {
  int winner;  // 0 = player 0
  int turns;
  bool timeout[2];
};

Result play(const std::string exe[2], const std::string env[2], const Track& tr, bool swapGrid) {
  Proc p[2];
  for (int t = 0; t < 2; ++t) p[t].start(exe[t], env[t]);
  std::string init = std::to_string(tr.laps) + "\n" + std::to_string(tr.n) + "\n";
  for (int i = 0; i < tr.n; ++i) init += std::to_string((int)tr.x[i]) + " " + std::to_string((int)tr.y[i]) + "\n";
  for (int t = 0; t < 2; ++t) p[t].send(init);
  State s;
  initState(s, tr, swapGrid);
  for (int k = 0; k < 4; ++k) {
    s.p[k].rad = std::atan2(tr.y[1 % tr.n] - s.p[k].y, tr.x[1 % tr.n] - s.p[k].x);
    s.p[k].angle = normAngle(s.p[k].rad / DEG);
  }
  Result res{-1, 0, {false, false}};
  for (int turn = 0; s.winner < 0 && turn < 600; ++turn) {
    double tx[4], ty[4];
    int th[4], kd[4];
    for (int t = 0; t < 2 && res.winner < 0; ++t) {
      std::string msg;
      for (int j = 0; j < 4; ++j) {
        const Pod& q = s.p[(2 * t + j) & 3];
        int ang = turn == 0 ? -1 : ((int)roundHalfUp(q.angle)) % 360;
        msg += std::to_string((long)q.x) + " " + std::to_string((long)q.y) + " " + std::to_string((long)q.vx) + " " +
               std::to_string((long)q.vy) + " " + std::to_string(ang) + " " + std::to_string(q.next) + "\n";
      }
      p[t].send(msg);
      for (int j = 0; j < 2; ++j) {
        std::string l;
        if (!p[t].line(l, turn == 0 ? 1500 : 400)) {
          res.timeout[t] = true;
          res.winner = 1 - t;
          break;
        }
        std::istringstream is(l);
        std::string w;
        int k = 2 * t + j;
        is >> tx[k] >> ty[k] >> w;
        kd[k] = w == "BOOST" ? K_BOOST : w == "SHIELD" ? K_SHIELD : K_THRUST;
        th[k] = kd[k] == K_THRUST ? std::max(0, std::min(MAX_THRUST, std::atoi(w.c_str()))) : MAX_THRUST;
      }
    }
    if (res.winner >= 0) break;
    for (int k = 0; k < 4; ++k) applyCommand(s, k, tx[k], ty[k], th[k], kd[k], turn == 0);
    step(s, tr);
    res.turns = turn + 1;
  }
  if (res.winner < 0) {
    res.winner = s.winner;
    if (res.winner < 0) {  // turn cap: the leader wins
      double a = std::max(progress(s.p[0], tr), progress(s.p[1], tr)), b = std::max(progress(s.p[2], tr), progress(s.p[3], tr));
      res.winner = a >= b ? 0 : 1;
    }
  }
  for (auto& q : p) q.stop();
  return res;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: arena BOT_A BOT_B GAMES [THREADS] [MAPS] [SEED]\n");
    return 1;
  }
  signal(SIGPIPE, SIG_IGN);
  std::string A = argv[1], B = argv[2];
  int games = std::atoi(argv[3]);
  int threads = argc > 4 ? std::atoi(argv[4]) : 4;
  std::vector<std::vector<int>> maps;
  if (argc > 5) {
    std::ifstream f(argv[5]);
    std::string l;
    while (std::getline(f, l)) {
      std::istringstream is(l);
      std::vector<int> v;
      int x;
      while (is >> x) v.push_back(x);
      if (v.size() >= 4) maps.push_back(v);
    }
  }
  uint64_t seed = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 1;
  const char* ea = std::getenv("ENV_A");
  const char* eb = std::getenv("ENV_B");
  std::atomic<int> next{0};
  std::mutex mu;
  int winsA = 0, done = 0, toA = 0, toB = 0;
  auto worker = [&]() {
    for (;;) {
      int g = next++;
      if (g >= games) return;
      Rng rng(seed * 1000003 + g / 2);
      Track tr;
      tr.laps = 3;
      const auto& m = maps[rng.below((int)maps.size())];
      tr.n = (int)m.size() / 2;
      for (int i = 0; i < tr.n; ++i) tr.x[i] = m[2 * i], tr.y[i] = m[2 * i + 1];
      bool aFirst = g % 2 == 0;  // A plays seat 0 in even games, seat 1 in odd ones, on the same map
      std::string exe[2] = {aFirst ? A : B, aFirst ? B : A};
      std::string env[2] = {aFirst ? (ea ? ea : "") : (eb ? eb : ""), aFirst ? (eb ? eb : "") : (ea ? ea : "")};
      Result r = play(exe, env, tr, false);
      bool aWon = (r.winner == 0) == aFirst;
      std::lock_guard<std::mutex> lk(mu);
      ++done;
      winsA += aWon;
      toA += r.timeout[aFirst ? 0 : 1];
      toB += r.timeout[aFirst ? 1 : 0];
      std::printf("game %d map %d cps seatA %d winner %s turns %d%s\n", g, tr.n, aFirst ? 0 : 1, aWon ? "A" : "B",
                  r.turns, r.timeout[0] || r.timeout[1] ? " (timeout)" : "");
      std::fflush(stdout);
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  double p = (double)winsA / std::max(1, done);
  double se = std::sqrt(p * (1 - p) / std::max(1, done));
  std::printf("A %d / %d = %.1f%% +- %.1f  (timeouts A %d B %d)\n", winsA, done, 100 * p, 100 * se, toA, toB);
}
