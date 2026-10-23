// Zero-sum matrix game solver: rows maximise A, columns minimise it.
// Regret matching+ with alternating updates and linearly weighted averages.
#pragma once
#include <algorithm>

namespace mpr {

constexpr int GAME_MAX = 8;

// A is m x n, row-major. Writes mixed strategies x (rows) and y (columns) and
// returns the game value x^T A y.
inline float solveGame(const float* A, int m, int n, float* x, float* y, int iters = 150) {
  // A saddle point (pure equilibrium) is common and cheap to detect.
  int bi = 0, bj = 0;
  float maxmin = -1e30f, minmax = 1e30f;
  for (int i = 0; i < m; ++i) {
    float mn = 1e30f;
    for (int j = 0; j < n; ++j) mn = std::min(mn, A[i * n + j]);
    if (mn > maxmin) maxmin = mn, bi = i;
  }
  for (int j = 0; j < n; ++j) {
    float mx = -1e30f;
    for (int i = 0; i < m; ++i) mx = std::max(mx, A[i * n + j]);
    if (mx < minmax) minmax = mx, bj = j;
  }
  if (maxmin >= minmax - 1e-6f) {
    std::fill(x, x + m, 0.f);
    std::fill(y, y + n, 0.f);
    x[bi] = 1;
    y[bj] = 1;
    return A[bi * n + bj];
  }
  float rx[GAME_MAX] = {}, ry[GAME_MAX] = {}, sx[GAME_MAX] = {}, sy[GAME_MAX] = {};
  float cx[GAME_MAX], cy[GAME_MAX], u[GAME_MAX];
  auto strat = [](const float* r, int k, float* out) {
    float s = 0;
    for (int i = 0; i < k; ++i) s += r[i];
    for (int i = 0; i < k; ++i) out[i] = s > 0 ? r[i] / s : 1.0f / k;
  };
  for (int t = 1; t <= iters; ++t) {
    strat(rx, m, cx);
    strat(ry, n, cy);
    float v = 0;
    for (int i = 0; i < m; ++i) {
      u[i] = 0;
      for (int j = 0; j < n; ++j) u[i] += A[i * n + j] * cy[j];
      v += cx[i] * u[i];
    }
    for (int i = 0; i < m; ++i) rx[i] = std::max(0.f, rx[i] + u[i] - v);
    strat(rx, m, cx);
    for (int i = 0; i < m; ++i) sx[i] += t * cx[i];
    float w = 0;
    for (int j = 0; j < n; ++j) {
      u[j] = 0;
      for (int i = 0; i < m; ++i) u[j] += cx[i] * A[i * n + j];
      w += cy[j] * u[j];
    }
    for (int j = 0; j < n; ++j) ry[j] = std::max(0.f, ry[j] + w - u[j]);
    strat(ry, n, cy);
    for (int j = 0; j < n; ++j) sy[j] += t * cy[j];
  }
  strat(sx, m, x);
  strat(sy, n, y);
  float val = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) val += x[i] * A[i * n + j] * y[j];
  return val;
}

}  // namespace mpr
