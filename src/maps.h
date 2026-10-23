// Track generation for the arena and the trainer.
#pragma once
#include <cmath>
#include <cstdint>

#include "engine.h"
#include "rng.h"

namespace mpr {

// The 13 Coders Strike Back layouts (via Agade's runner arena / robostac).
inline const int kKnownMaps[13][7][2] = {
    {{12460, 1350}, {10540, 5980}, {3580, 5180}, {13580, 7600}},
    {{3600, 5280}, {13840, 5080}, {10680, 2280}, {8700, 7460}, {7200, 2160}},
    {{4560, 2180}, {7350, 4940}, {3320, 7230}, {14580, 7700}, {10560, 5060}, {13100, 2320}},
    {{5010, 5260}, {11480, 6080}, {9100, 1840}},
    {{14660, 1410}, {3450, 7220}, {9420, 7240}, {5970, 4240}},
    {{3640, 4420}, {8000, 7900}, {13300, 5540}, {9560, 1400}},
    {{4100, 7420}, {13500, 2340}, {12940, 7220}, {5640, 2580}},
    {{14520, 7780}, {6320, 4290}, {7800, 860}, {7660, 5970}, {3140, 7540}, {9520, 4380}},
    {{10040, 5970}, {13920, 1940}, {8020, 3260}, {2670, 7020}},
    {{7500, 6940}, {6000, 5360}, {11300, 2820}},
    {{4060, 4660}, {13040, 1900}, {6560, 7840}, {7480, 1360}, {12700, 7100}},
    {{3020, 5190}, {6280, 7760}, {14100, 7760}, {13880, 1220}, {10240, 4920}, {6100, 2200}},
    {{10323, 3366}, {11203, 5425}, {7259, 6656}, {5425, 2838}},
};
inline const int kKnownCounts[13] = {4, 5, 6, 3, 4, 4, 4, 6, 4, 3, 5, 6, 4};

// A known layout, jittered by up to 30 units, rotated to a random start and
// sometimes driven in reverse.
inline Track knownTrack(Rng& r) {
  int m = r.below(13);
  Track t;
  t.n = kKnownCounts[m];
  int start = r.below(t.n);
  bool rev = r.below(2);
  for (int i = 0; i < t.n; ++i) {
    int j = rev ? (start - i + t.n) % t.n : (start + i) % t.n;
    t.x[i] = kKnownMaps[m][j][0] + r.below(61) - 30;
    t.y[i] = kKnownMaps[m][j][1] + r.below(61) - 30;
  }
  return t;
}

// A random layout with 2..8 checkpoints (the statement's range), all at
// least 2500 apart and 1000 from the map edge.
inline Track randomTrack(Rng& r, int minN = 2, int maxN = 8) {
  Track t;
  for (;;) {
    t.n = minN + r.below(maxN - minN + 1);
    if (t.n == 2 && r.below(4)) continue;  // keep 2-checkpoint tracks rare
    bool ok = true;
    for (int i = 0; i < t.n && ok; ++i) {
      int tries = 0;
      for (;; ++tries) {
        t.x[i] = 1000 + r.below(14001);
        t.y[i] = 1000 + r.below(7001);
        bool far = true;
        for (int j = 0; j < i; ++j)
          if (std::hypot(t.x[i] - t.x[j], t.y[i] - t.y[j]) < 2500) far = false;
        if (far) break;
        if (tries > 200) {
          ok = false;
          break;
        }
      }
    }
    if (ok) return t;
  }
}

inline Track anyTrack(Rng& r) { return r.below(10) < 6 ? knownTrack(r) : randomTrack(r); }

}  // namespace mpr
