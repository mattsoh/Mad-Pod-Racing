"""Loss analysis over reconstructed battles: how each game ended and where the race was lost."""
import glob, math, sys, collections, pickle, os
from games import load_game

D = sys.argv[1]
cache = os.path.join(D, "..", "games.pkl")
if os.path.exists(cache):
    G = pickle.load(open(cache, "rb"))
else:
    G = [load_game(f) for f in sorted(glob.glob(D + "/*.json"))]
    pickle.dump(G, open(cache, "wb"))


def prog(p, cps):
    nx, ny = cps[p["next"]]
    return p["passed"] * 50000 - math.hypot(nx - p["x"], ny - p["y"])


def lead_curve(g):
    """Per turn: our best pod's progress minus theirs, in checkpoints (approx)."""
    cps, me = g["track"], g["me"]
    out = []
    for t in g["turns"]:
        P = t["pods"]
        mine = max(P[2 * me]["passed"], P[2 * me + 1]["passed"])
        theirs = max(P[2 - 2 * me]["passed"], P[3 - 2 * me]["passed"])
        out.append(mine - theirs)
    return out


rows = []
for g in G:
    if g["sim_winner"] < 0:
        continue
    T = g["turns"]
    last = T[-1]
    me = g["me"]
    to_me, to_op = last["to"][me], last["to"][1 - me]
    kind = "finish"
    if g["sim_turns"] and to_me <= 1 and not g["won"]:
        kind = "our_timeout"
    elif to_op <= 1 and g["won"]:
        kind = "their_timeout"
    lc = lead_curve(g)
    total = len(g["track"]) * g["laps"]
    # turn at which each side's leader reached half of the race
    def first_turn(cond):
        for i, v in enumerate(T):
            if cond(v):
                return i
        return None
    me_half = first_turn(lambda t: max(t["pods"][2 * me]["passed"], t["pods"][2 * me + 1]["passed"]) >= total // 2)
    op_half = first_turn(lambda t: max(t["pods"][2 - 2 * me]["passed"], t["pods"][3 - 2 * me]["passed"]) >= total // 2)
    shields_me = sum(1 for t in T for j in range(2) if t.get("cmd") and t["cmd"][2 * me + j][2] == "SHIELD")
    shields_op = sum(1 for t in T for j in range(2) if t.get("cmd") and t["cmd"][2 - 2 * me + j][2] == "SHIELD")
    rows.append(dict(id=g["game_id"], opp=g["opp"], won=g["won"], kind=kind, turns=len(T), total=total,
                     lead_max=max(lc), lead_min=min(lc), lead_end=lc[-1], me_half=me_half, op_half=op_half,
                     sh_me=shields_me, sh_op=shields_op, seat=me, ncp=len(g["track"])))

W = [r for r in rows if r["won"]]
L = [r for r in rows if not r["won"]]
print(f"{len(rows)} games, {len(W)} won, {len(L)} lost")
print("loss kinds:", collections.Counter(r["kind"] for r in L))
print("win kinds: ", collections.Counter(r["kind"] for r in W))
print("lost games: end lead (cps):", collections.Counter(r["lead_end"] for r in L if r["kind"] == "finish"))
print("lost games where we led by >=1 cp at some point:", sum(r["lead_max"] >= 1 for r in L), "of", len(L))
print("lost games where we led by >=2 cp:", sum(r["lead_max"] >= 2 for r in L))
for nm, S in (("won", W), ("lost", L)):
    hh = [(r["op_half"] or 999) - (r["me_half"] or 999) for r in S if r["me_half"] and r["op_half"]]
    print(f"{nm}: half-race turn advantage mean {sum(hh)/max(1,len(hh)):.1f} n={len(hh)}; mean turns {sum(r['turns'] for r in S)/len(S):.0f}"
          f"; shields me {sum(r['sh_me'] for r in S)/len(S):.1f} opp {sum(r['sh_op'] for r in S)/len(S):.1f}")
print("by seat: ", {s: f"{sum(r['won'] for r in rows if r['seat']==s)}/{sum(1 for r in rows if r['seat']==s)}" for s in (0, 1)})
print("by #cps: ", {n: f"{sum(r['won'] for r in rows if r['ncp']==n)}/{sum(1 for r in rows if r['ncp']==n)}" for n in sorted(set(r['ncp'] for r in rows))})
top = ["Discourage", "sffsefsdfaf", "DaNinja", "JackieChun", "RoboStac", "naab", "jBigman", "MSmits", "IGGY"]
for o in top:
    S = [r for r in rows if r["opp"] == o]
    if not S: continue
    print(f"{o:12s} {sum(r['won'] for r in S)}-{sum(not r['won'] for r in S)} kinds {dict(collections.Counter((r['won'], r['kind']) for r in S))}"
          f" led>=1 in losses {sum(r['lead_max']>=1 for r in S if not r['won'])}  sh me/op {sum(r['sh_me'] for r in S)/len(S):.1f}/{sum(r['sh_op'] for r in S)/len(S):.1f}")
