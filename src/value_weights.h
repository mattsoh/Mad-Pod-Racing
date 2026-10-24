// No value network embedded (valueW has no effect). `python3 tools/bundle.py ... --value nets/valueN_weights.h`
// embeds one; tools/train_value.py writes them.
#pragma once
inline const char* NN_VALUE = "";
