"""Inline the bot's local headers into one file for the CodinGame IDE.

    python3 tools/bundle.py src/bot.cpp dist/mad_pod_bot.cpp [--weights nets/duel_weights.h]

Each `#include "x.h"` is replaced by that header (once). With --weights, that
file stands in for src/weights.h. Comments are stripped only if the result
would exceed CodinGame's 100,000-character limit.
"""
import argparse
import os
import re
import sys

LIMIT = 100_000
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')


def inline(path, seen, weights, value=None):
    path = os.path.normpath(path)
    if os.path.basename(path) == "weights.h" and weights:
        path = weights
    if os.path.basename(path) == "value_weights.h" and value:
        path = value
    if path in seen:
        return []
    seen.add(path)
    out = []
    skipping = False
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        # Code the submitted bot never runs (e.g. the old duel network) is marked and left out of the bundle.
        if s.startswith("// @bundle-skip-begin"):
            skipping = True
            continue
        if s.startswith("// @bundle-skip-end"):
            skipping = False
            continue
        if skipping:
            continue
        if s.startswith("// @bundle-only "):
            line = line.replace("// @bundle-only ", "", 1)
        m = INCLUDE.match(line)
        if m:
            out += inline(os.path.join(os.path.dirname(path), m.group(1)), seen, weights, value)
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return out


def strip_comments(lines):
    out = []
    for line in lines:
        s = line.strip()
        if not s or (s.startswith("//") and '"' not in s):
            continue
        if "//" in line and '"' not in line and "'" not in line:  # trailing comment, no literal on the line
            line = line[:line.index("//")].rstrip() + "\n"
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights", help="weights header to embed instead of src/weights.h")
    ap.add_argument("--value", help="value-network header to embed instead of src/value_weights.h")
    a = ap.parse_args()
    lines = inline(a.src, set(), a.weights and os.path.normpath(a.weights), a.value and os.path.normpath(a.value))
    text = "".join(lines)
    if len(text) > LIMIT:
        text = "".join(strip_comments(lines))
    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    with open(a.dst, "w", encoding="utf-8") as f:
        f.write(text)
    status = "OK" if len(text) <= LIMIT else "TOO LARGE for CodinGame"
    print(f"{a.dst}: {len(text):,} characters ({status}, limit {LIMIT:,})")
    if len(text) > LIMIT:
        sys.exit(1)


if __name__ == "__main__":
    main()
