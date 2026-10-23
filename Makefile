CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-result
BUILD    := build

all: $(BUILD)/bot $(BUILD)/arena $(BUILD)/engine_cli $(BUILD)/train $(BUILD)/bc $(BUILD)/predict_eval $(BUILD)/replay_check $(BUILD)/cglog_states

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/bot: src/*.h src/bot.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/bot.cpp

$(BUILD)/arena: src/*.h tools/arena.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ tools/arena.cpp

$(BUILD)/engine_cli: src/engine.h tools/engine_cli.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tools/engine_cli.cpp

$(BUILD)/train: src/*.h tools/nn_train.h tools/train.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -O3 -march=native -fopenmp-simd -pthread -o $@ tools/train.cpp

$(BUILD)/bc: src/*.h tools/nn_train.h tools/bc.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -O3 -march=native -fopenmp-simd -pthread -o $@ tools/bc.cpp

$(BUILD)/predict_eval: src/*.h tools/nn_train.h tools/predict_eval.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ tools/predict_eval.cpp

$(BUILD)/replay_check: src/engine.h tools/replay_check.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tools/replay_check.cpp

$(BUILD)/cglog_states: src/engine.h tools/cglog_states.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tools/cglog_states.cpp

# One self-contained file to paste into the CodinGame IDE (language: C++), with
# the imitation network embedded (hybrid mode: it predicts the opponent and
# seeds the search).
WEIGHTS ?= nets/bc160_t15_weights.h
bundle: | $(BUILD)
	python3 tools/bundle.py src/bot.cpp dist/mad_pod_bot.cpp --weights $(WEIGHTS)

# The same bot without a network (pure search).
bundle-search: | $(BUILD)
	python3 tools/bundle.py src/bot.cpp dist/mad_pod_bot_search.cpp

# Bot compiled the way CodinGame does (no -O flag; the file's pragmas apply).
$(BUILD)/bot_cg: bundle
	$(CXX) -std=c++17 -o $@ dist/mad_pod_bot.cpp

# Diff the engine against robostac's replay-validated Go referee (needs git and Go).
# Re-simulate every downloaded real game (data/games.txt) and count mismatches.
check-replays: $(BUILD)/bc
	$(BUILD)/bc --data data/games.txt --epochs 0

crosscheck: $(BUILD)/engine_cli
	test -d ref/robostac || git clone --depth 1 https://github.com/robostac/coders-strike-back-referee.git ref/robostac
	cd ref/robostac && go build -o csbref csbref.go
	python3 tools/crosscheck.py --ref ref/robostac/csbref --ours $(BUILD)/engine_cli --games 300

clean:
	rm -rf $(BUILD) dist

.PHONY: all bundle bundle-search check-replays crosscheck clean
