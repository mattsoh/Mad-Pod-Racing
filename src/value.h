// Learned value of a whole 4-pod situation: how the race lead tends to change over the next 30 turns
// (in checkpoints), trained on top-15 bots' games (tools/value_data.cpp, tools/train_value.py). The search
// adds valueW * 50000 * V at the horizon, so multi-pod tactics that gain ground (marking the enemy blocker,
// double-teaming its runner, escorting) are valued wherever they occur instead of by fixed role terms.
#pragma once
#include "features.h"
#include "nn.h"
#include "value_weights.h"

namespace mpr {

inline const MLP& valueNet() {
  static MLP m;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    if (NN_VALUE[0]) dequantize(z85Decode(NN_VALUE), m);
  }
  return m;
}
inline bool valueAvailable() { return valueNet().inputs() == N_FULL && valueNet().outputs() == 1; }

// Value for `team`, seen from its runner (the pod with more progress).
inline double valueOf(const State& s, const Track& tr, int team) {
  int a = 2 * team, run = progress(s.p[a], tr) >= progress(s.p[a + 1], tr) ? a : a + 1;
  float f[N_FULL], v;
  fullFeatures(s, tr, run, f);
  valueNet().forward(f, &v);
  return v;
}

}  // namespace mpr
