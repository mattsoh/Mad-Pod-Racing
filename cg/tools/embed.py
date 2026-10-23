"""Writes a copy of the bot with another network: python3 embed.py BOT.cpp WEIGHTS.txt OUT.cpp [-D NAME=VALUE ...]"""
import re, sys
src, wts, out = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
blob = open(wts).read().strip()
a = s.index("inline const char* NN_WEIGHTS =")
b = s.index(";", a)
lines = ['    "' + blob[i:i + 120].replace("?", "\\?") + '"' for i in range(0, len(blob), 120)]
s = s[:a] + "inline const char* NN_WEIGHTS =\n" + "\n".join(lines) + s[b:]
open(out, "w").write(s)
print(f"{out}: {len(s)} chars (CodinGame limit 100000)")
