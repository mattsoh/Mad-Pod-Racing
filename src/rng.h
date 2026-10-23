// Small, fast, seedable RNG shared by the bot, arena and trainer.
#pragma once
#include <cstdint>

namespace mpr {

// xorshift64*
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 88172645463325252ull) : s(seed ? seed : 1) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ull;
  }
  int below(int n) { return (int)((next() >> 33) % (uint64_t)n); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double range(double a, double b) { return a + (b - a) * uniform(); }
};

}  // namespace mpr
