#!/bin/bash
# try_net.sh NAME DATASET [train_bc args...]: train nets/NAME, build bots/NAME.cpp with it, print its horizon errors
# on the games in $TESTDIR. BASE: bot to put the network in (bots/lm_base.cpp for --last networks).
set -e
cd "$(dirname "$0")/.."
name=$1; ds=$2; shift 2
base=${BASE:-bots/live_41511466.cpp}
python3 tools/train_bc.py "$ds" nets/$name.txt "$@" | grep -E "init network|quantized" || true
python3 tools/embed.py $base nets/$name.txt bots/$name.cpp > /dev/null
g++ -O2 -std=c++17 -DBOT_SOURCE="\"../bots/$name.cpp\"" -o build/reval_$name src/rollout_eval.cpp
cd tools && python3 reval.py all "$TESTDIR" ../build/reval_$name | grep network
