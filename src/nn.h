// Small dense network (leaky ReLU hidden layers, linear output), plus the
// int8 + Z85 text encoding used to carry its weights inside the source file.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mpr {

struct Dense {
  int in = 0, out = 0;
  std::vector<float> W, b;  // W[o * in + i]
};

inline float leaky(float v) { return v > 0 ? v : 0.01f * v; }

struct MLP {
  std::vector<Dense> L;

  int inputs() const { return L.empty() ? 0 : L.front().in; }
  int outputs() const { return L.empty() ? 0 : L.back().out; }

  void init(const std::vector<int>& sizes, uint64_t seed) {
    L.clear();
    uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto rnd = [&]() {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      return ((s >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1;
    };
    for (size_t k = 0; k + 1 < sizes.size(); ++k) {
      Dense d;
      d.in = sizes[k];
      d.out = sizes[k + 1];
      d.W.resize((size_t)d.in * d.out);
      d.b.assign(d.out, 0.f);
      double lim = std::sqrt(6.0 / d.in);  // He-uniform
      for (auto& w : d.W) w = (float)(rnd() * lim);
      L.push_back(d);
    }
  }

  // Thread-safe: scratch space is per thread.
  void forward(const float* x, float* y) const {
    thread_local std::vector<float> bufA, bufB;
    size_t width = 0;
    for (auto& d : L) width = std::max(width, (size_t)d.out);
    if (bufA.size() < width) bufA.resize(width), bufB.resize(width);
    const float* in = x;
    for (size_t k = 0; k < L.size(); ++k) {
      const Dense& d = L[k];
      bool last = k + 1 == L.size();
      float* out = last ? y : (k & 1 ? bufB.data() : bufA.data());
      for (int o = 0; o < d.out; ++o) {
        const float* w = &d.W[(size_t)o * d.in];
        float acc = 0;
        for (int i = 0; i < d.in; ++i) acc += w[i] * in[i];
        acc += d.b[o];
        out[o] = last ? acc : leaky(acc);
      }
      in = out;
    }
  }
};

// ---- int8 quantisation ------------------------------------------------------
// Blob: "MPR1", u8 layers, (u16 in, u16 out) per layer, then per layer:
// per output row a float32 scale and `in` int8 weights, then float32 biases.

inline std::string quantize(const MLP& m) {
  std::string b = "MPR1";
  b.push_back((char)m.L.size());
  auto put16 = [&](int v) { b.push_back((char)(v & 255)), b.push_back((char)(v >> 8)); };
  auto putf = [&](float f) {
    char c[4];
    std::memcpy(c, &f, 4);
    b.append(c, 4);
  };
  for (auto& d : m.L) put16(d.in), put16(d.out);
  for (auto& d : m.L) {
    for (int o = 0; o < d.out; ++o) {
      float mx = 0;
      for (int i = 0; i < d.in; ++i) mx = std::max(mx, std::fabs(d.W[(size_t)o * d.in + i]));
      float sc = mx > 0 ? mx / 127.f : 1.f;
      putf(sc);
      for (int i = 0; i < d.in; ++i) {
        int q = (int)std::lround(d.W[(size_t)o * d.in + i] / sc);
        b.push_back((char)(int8_t)std::max(-127, std::min(127, q)));
      }
    }
    for (int o = 0; o < d.out; ++o) putf(d.b[o]);
  }
  return b;
}

inline bool dequantize(const std::string& b, MLP& m) {
  if (b.size() < 5 || b.compare(0, 4, "MPR1") != 0) return false;
  size_t p = 4;
  int n = (uint8_t)b[p++];
  m.L.assign(n, Dense());
  auto get16 = [&]() {
    int v = (uint8_t)b[p] | ((uint8_t)b[p + 1] << 8);
    p += 2;
    return v;
  };
  auto getf = [&]() {
    float f;
    std::memcpy(&f, b.data() + p, 4);
    p += 4;
    return f;
  };
  for (auto& d : m.L) {
    if (p + 4 > b.size()) return false;
    d.in = get16();
    d.out = get16();
    d.W.resize((size_t)d.in * d.out);
    d.b.resize(d.out);
  }
  for (auto& d : m.L) {
    if (p + (size_t)d.out * (4 + d.in) + 4 * (size_t)d.out > b.size()) return false;
    for (int o = 0; o < d.out; ++o) {
      float sc = getf();
      for (int i = 0; i < d.in; ++i) d.W[(size_t)o * d.in + i] = (int8_t)b[p++] * sc;
    }
    for (int o = 0; o < d.out; ++o) d.b[o] = getf();
  }
  return true;
}

// ---- Z85 (ZeroMQ base85): no quotes or backslashes, safe in a C string ------

inline const char* kZ85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

inline std::string z85Encode(std::string bytes) {
  while (bytes.size() % 4) bytes.push_back(0);
  std::string out;
  for (size_t i = 0; i < bytes.size(); i += 4) {
    uint32_t v = ((uint32_t)(uint8_t)bytes[i] << 24) | ((uint32_t)(uint8_t)bytes[i + 1] << 16) |
                 ((uint32_t)(uint8_t)bytes[i + 2] << 8) | (uint8_t)bytes[i + 3];
    char c[5];
    for (int k = 4; k >= 0; --k) c[k] = kZ85[v % 85], v /= 85;
    out.append(c, 5);
  }
  return out;
}

inline std::string z85Decode(const char* s) {
  int rev[128];
  for (int i = 0; i < 128; ++i) rev[i] = 0;
  for (int i = 0; i < 85; ++i) rev[(int)kZ85[i]] = i;
  std::string out;
  size_t n = std::strlen(s);
  for (size_t i = 0; i + 5 <= n; i += 5) {
    uint32_t v = 0;
    for (int k = 0; k < 5; ++k) v = v * 85 + rev[(int)s[i + k] & 127];
    out.push_back((char)(v >> 24));
    out.push_back((char)(v >> 16));
    out.push_back((char)(v >> 8));
    out.push_back((char)v);
  }
  return out;
}

}  // namespace mpr
