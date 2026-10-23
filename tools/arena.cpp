// Local arena: plays two bot programs against each other with the real
// protocol, using the same engine as the bot.
//
//   build/arena --games 100 --time-ms 20 "build/bot" "build/bot_old"
//
// Each seed is played twice with seats swapped. Bots get MPR_TIME_MS in their
// environment so they can think less than the 75 ms CodinGame allows.
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/engine.h"
#include "../src/maps.h"

extern char** environ;
using namespace mpr;
using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Proc {
  pid_t pid = -1;
  int toBot = -1, fromBot = -1;
  std::string buf;
  bool dead = false;

  bool start(const std::string& cmd, const std::vector<std::string>& extraEnv, const std::string& errPath) {
    int in[2], out[2];
    if (pipe(in) || pipe(out)) return false;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, errPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_addclose(&fa, in[1]);
    posix_spawn_file_actions_addclose(&fa, out[0]);
    std::vector<std::string> envs;
    for (char** e = environ; *e; ++e) envs.push_back(*e);
    for (auto& e : extraEnv) envs.push_back(e);
    std::vector<char*> envp;
    for (auto& e : envs) envp.push_back(e.data());
    envp.push_back(nullptr);
    std::string c = cmd;
    char* argv[] = {(char*)"/bin/sh", (char*)"-c", c.data(), nullptr};
    int rc = posix_spawn(&pid, "/bin/sh", &fa, nullptr, argv, envp.data());
    posix_spawn_file_actions_destroy(&fa);
    close(in[0]);
    close(out[1]);
    toBot = in[1];
    fromBot = out[0];
    fcntl(fromBot, F_SETFL, O_NONBLOCK);
    fcntl(toBot, F_SETFD, FD_CLOEXEC);
    fcntl(fromBot, F_SETFD, FD_CLOEXEC);
    return rc == 0;
  }
  bool send(const std::string& s) {
    if (dead) return false;
    size_t off = 0;
    while (off < s.size()) {
      ssize_t w = write(toBot, s.data() + off, s.size() - off);
      if (w <= 0) return dead = true, false;
      off += (size_t)w;
    }
    return true;
  }
  int lines() const {
    int c = 0;
    for (char ch : buf) c += ch == '\n';
    return c;
  }
  std::string popLine() {
    size_t p = buf.find('\n');
    std::string l = buf.substr(0, p);
    buf.erase(0, p + 1);
    if (!l.empty() && l.back() == '\r') l.pop_back();
    return l;
  }
  void stop() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      pid = -1;
    }
    if (toBot >= 0) close(toBot), toBot = -1;
    if (fromBot >= 0) close(fromBot), fromBot = -1;
  }
};

// Waits until every live bot has `want` lines or its deadline passes.
// Fills the time each bot took to deliver its last line.
static void collect(Proc* p[2], int want, double limitMs, Clock::time_point sent, double took[2], bool late[2]) {
  for (int i = 0; i < 2; ++i) took[i] = 0, late[i] = false;
  for (;;) {
    pollfd fds[2];
    int map[2], nf = 0;
    for (int i = 0; i < 2; ++i)
      if (!p[i]->dead && p[i]->lines() < want) fds[nf] = {p[i]->fromBot, POLLIN, 0}, map[nf++] = i;
    if (!nf) return;
    double left = limitMs - msSince(sent);
    if (left <= 0) {
      for (int f = 0; f < nf; ++f) late[map[f]] = true;
      return;
    }
    int rc = poll(fds, nf, (int)std::ceil(left));
    if (rc < 0) continue;
    for (int f = 0; f < nf; ++f) {
      if (!(fds[f].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      Proc* q = p[map[f]];
      char tmp[4096];
      ssize_t r = read(q->fromBot, tmp, sizeof tmp);
      if (r > 0) {
        q->buf.append(tmp, (size_t)r);
        if (q->lines() >= want) took[map[f]] = msSince(sent);
      } else if (r == 0) {
        q->dead = true;
      }
    }
  }
}

static bool parseCommand(const std::string& line, double& x, double& y, int& thrust, int& kind) {
  std::istringstream in(line);
  std::string tok;
  if (!(in >> x >> y >> tok)) return false;
  kind = K_THRUST;
  thrust = 0;
  if (tok == "SHIELD") kind = K_SHIELD;
  else if (tok == "BOOST") kind = K_BOOST;
  else {
    char* end = nullptr;
    long v = std::strtol(tok.c_str(), &end, 10);
    if (*end || v < 0 || v > MAX_THRUST) return false;
    thrust = (int)v;
  }
  return true;
}

struct Options {
  std::string bot[2];
  int games = 50, threads = 1;
  uint64_t seed = 1;
  double timeMs = 20, limitMs = -1, firstLimitMs = 2000;
  std::string maps = "mix", replayDir, logDir = "/dev/null";
  bool verbose = false;
};

struct Result {
  int winnerBot = -1;  // 0 = first bot, 1 = second bot
  int turns = 0;
  std::string reason;
  double maxMs[2] = {0, 0};
};

static Result playGame(const Options& o, int g) {
  Rng r(o.seed * 1000003 + (uint64_t)(g / 2) * 7919 + 17);
  Track tr = o.maps == "known" ? knownTrack(r) : o.maps == "random" ? randomTrack(r) : anyTrack(r);
  int seatOf[2] = {g & 1, 1 - (g & 1)};  // seatOf[bot] = team
  Proc procs[2];
  Proc* team[2];
  std::vector<std::string> env = {"MPR_TIME_MS=" + std::to_string(o.timeMs)};
  for (int b = 0; b < 2; ++b) {
    std::string err = o.logDir == "/dev/null" ? o.logDir : o.logDir + "/g" + std::to_string(g) + "_b" + std::to_string(b) + ".log";
    procs[b].start(o.bot[b], env, err);
    team[seatOf[b]] = &procs[b];
  }
  Result res;
  State s;
  initState(s, tr);
  std::ostringstream init;
  init << tr.laps << '\n' << tr.n << '\n';
  for (int i = 0; i < tr.n; ++i) init << (int)tr.x[i] << ' ' << (int)tr.y[i] << '\n';
  for (int t = 0; t < 2; ++t) team[t]->send(init.str());

  std::ostringstream rep;
  rep << "{\"laps\":" << tr.laps << ",\"checkpoints\":[";
  for (int i = 0; i < tr.n; ++i) rep << (i ? "," : "") << '[' << (int)tr.x[i] << ',' << (int)tr.y[i] << ']';
  rep << "],\"bots\":[\"" << o.bot[seatOf[0] == 0 ? 0 : 1] << "\",\"" << o.bot[seatOf[0] == 0 ? 1 : 0]
      << "\"],\"frames\":[";
  auto frame = [&](const std::string* msgs) {
    if (o.replayDir.empty()) return;
    rep << (s.turn ? "," : "") << "{\"pods\":[";
    for (int k = 0; k < 4; ++k) {
      const Pod& q = s.p[k];
      rep << (k ? "," : "") << '[' << (int)q.x << ',' << (int)q.y << ',' << (int)q.vx << ',' << (int)q.vy << ','
          << ((int)roundHalfUp(q.angle)) % 360 << ',' << q.next << ',' << q.passed << ',' << q.shield << ']';
    }
    rep << "],\"timeout\":[" << s.timeout[0] << ',' << s.timeout[1] << "],\"msg\":[";
    for (int k = 0; k < 4; ++k) {
      std::string m = msgs ? msgs[k] : "";
      for (char& c : m)
        if (c == '"' || c == '\\') c = ' ';
      rep << (k ? "," : "") << '"' << m << '"';
    }
    rep << "]}";
  };
  frame(nullptr);

  int loser = -1;  // team
  while (s.winner < 0 && loser < 0 && s.turn < 500) {
    Clock::time_point sent = Clock::now();
    for (int t = 0; t < 2; ++t) {
      std::ostringstream in;
      int order[4] = {2 * t, 2 * t + 1, 2 - 2 * t, 3 - 2 * t};
      for (int k : order) {
        const Pod& q = s.p[k];
        int ang = s.turn == 0 ? -1 : ((int)roundHalfUp(q.angle)) % 360;
        in << (int)q.x << ' ' << (int)q.y << ' ' << (int)q.vx << ' ' << (int)q.vy << ' ' << ang << ' ' << q.next << '\n';
      }
      team[t]->send(in.str());
    }
    double limit = s.turn == 0 ? o.firstLimitMs : o.limitMs;
    double took[2];
    bool late[2];
    collect(team, 2, limit, sent, took, late);
    std::string msgs[4];
    for (int t = 0; t < 2 && loser < 0; ++t) {
      int b = seatOf[0] == t ? 0 : 1;
      if (s.turn > 0) res.maxMs[b] = std::max(res.maxMs[b], took[t]);
      if (late[t] || team[t]->dead || team[t]->lines() < 2) {
        loser = t;
        res.reason = team[t]->dead ? "crash" : "timeout";
        break;
      }
      for (int j = 0; j < 2; ++j) {
        std::string line = team[t]->popLine();
        double x, y;
        int thrust, kind;
        if (!parseCommand(line, x, y, thrust, kind)) {
          loser = t;
          res.reason = "invalid output: " + line;
          break;
        }
        std::istringstream rest(line);
        std::string a, b2, c, m;
        rest >> a >> b2 >> c;
        std::getline(rest, m);
        msgs[2 * t + j] = m.size() > 1 ? m.substr(1) : "";
        applyCommand(s, 2 * t + j, x, y, thrust, kind, s.turn == 0);
      }
    }
    if (loser >= 0) break;
    step(s, tr);
    frame(msgs);
  }
  int winTeam;
  if (loser >= 0) winTeam = 1 - loser;
  else if (s.winner >= 0) {
    winTeam = s.winner;
    res.reason = s.timeout[0] <= 0 || s.timeout[1] <= 0 ? "checkpoint timeout" : "finished";
  } else {
    double best = -1e18;
    winTeam = 0;
    for (int k = 0; k < 4; ++k)
      if (progress(s.p[k], tr) > best) best = progress(s.p[k], tr), winTeam = k >> 1;
    res.reason = "turn limit";
  }
  res.winnerBot = seatOf[0] == winTeam ? 0 : 1;
  res.turns = s.turn;
  for (auto& p : procs) p.stop();
  if (!o.replayDir.empty()) {
    rep << "],\"winner\":" << winTeam << ",\"reason\":\"" << res.reason << "\"}";
    std::ofstream(o.replayDir + "/game" + std::to_string(g) + ".json") << rep.str();
  }
  return res;
}

static void usage() {
  fprintf(stderr,
          "usage: arena [options] BOT_A BOT_B\n"
          "  --games N        games to play (seats swap every game; default 50)\n"
          "  --seed S         base seed (default 1)\n"
          "  --time-ms T      think time given to bots via MPR_TIME_MS (default 20)\n"
          "  --limit-ms L     per-turn response limit (default 3*T + 30)\n"
          "  --maps M         known | random | mix (default mix)\n"
          "  --per-pod-boost  each pod gets its own boost (already the default; real games confirm it)\n"
          "  --replays DIR    write one JSON replay per game\n"
          "  --logs DIR       keep each bot's stderr\n"
          "  -j N             games in parallel (default 1)\n"
          "  -v               print every game\n");
}

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  Options o;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(), exit(2);
      return argv[++i];
    };
    if (a == "--games") o.games = std::stoi(val());
    else if (a == "--seed") o.seed = std::stoull(val());
    else if (a == "--time-ms") o.timeMs = std::stod(val());
    else if (a == "--limit-ms") o.limitMs = std::stod(val());
    else if (a == "--maps") o.maps = val();
    else if (a == "--per-pod-boost") g_sharedBoost = false;
    else if (a == "--replays") o.replayDir = val();
    else if (a == "--logs") o.logDir = val();
    else if (a == "-j") o.threads = std::stoi(val());
    else if (a == "-v") o.verbose = true;
    else if (a == "-h" || a == "--help") return usage(), 0;
    else pos.push_back(a);
  }
  if (pos.size() != 2) return usage(), 2;
  o.bot[0] = pos[0];
  o.bot[1] = pos[1];
  if (o.limitMs < 0) o.limitMs = 3 * o.timeMs + 30;

  std::atomic<int> nextGame{0};
  std::mutex mu;
  int wins[2] = {0, 0}, done = 0;
  long turns = 0;
  double maxMs[2] = {0, 0};
  std::vector<std::string> notes;
  auto worker = [&]() {
    for (int g; (g = nextGame++) < o.games;) {
      Result r = playGame(o, g);
      std::lock_guard<std::mutex> lk(mu);
      wins[r.winnerBot]++;
      done++;
      turns += r.turns;
      for (int b = 0; b < 2; ++b) maxMs[b] = std::max(maxMs[b], r.maxMs[b]);
      if (r.reason != "finished" && r.reason != "checkpoint timeout" && notes.size() < 10)
        notes.push_back("game " + std::to_string(g) + ": bot " + "AB"[r.winnerBot] + " won by " + r.reason);
      if (o.verbose)
        printf("game %3d  winner %c  turns %3d  %s\n", g, "AB"[r.winnerBot], r.turns, r.reason.c_str());
      else if (done % 10 == 0)
        fprintf(stderr, "\r%d/%d  A %d - B %d", done, o.games, wins[0], wins[1]);
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < std::max(1, o.threads); ++i) th.emplace_back(worker);
  for (auto& t : th) t.join();
  fprintf(stderr, "\n");

  double n = done, p = wins[0] / n, z = 1.96;
  double centre = (p + z * z / (2 * n)) / (1 + z * z / n);
  double half = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n);
  printf("A: %s\nB: %s\n", o.bot[0].c_str(), o.bot[1].c_str());
  printf("games %d   A wins %d   B wins %d   A win rate %.1f%% (95%% CI %.1f-%.1f%%)\n", done, wins[0], wins[1],
         100 * p, std::max(0.0, 100 * (centre - half)), std::min(100.0, 100 * (centre + half)));
  printf("avg turns %.1f   slowest turn: A %.1f ms, B %.1f ms (limit %.0f ms)\n", turns / n, maxMs[0], maxMs[1],
         o.limitMs);
  for (auto& s : notes) printf("  %s\n", s.c_str());
  return 0;
}
