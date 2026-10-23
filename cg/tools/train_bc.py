"""Trains the bot's opponent model (behaviour cloning of real Legend opponents) and writes it as the bot's
NN_WEIGHTS blob.

    python3 train_bc.py DATASET.npz OUT.txt [--hidden 160] [--epochs 40] [--who opp|all]

The network matches the bot's MLP: 44 -> H -> H -> 7, leaky ReLU (0.01); outputs 3 rotation logits
(-18, 0, +18 degrees) and 4 kind logits (thrust 0, thrust 200, SHIELD, BOOST). Games are split 85/15 for the
held-out score, which is compared with the network embedded in the bot.
"""
import argparse, struct
import numpy as np
import torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--hidden", type=int, default=160)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--who", default="opp")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mirror", action="store_true", help="add left-right mirrored copies of the training samples")
ap.add_argument("--mask-reload", action="store_true")
ap.add_argument("--last", action="store_true", help="4 more inputs: last turn's rotation, thrust, touched flag, masked flag")
ap.add_argument("--last-drop", type=float, default=0.5, help="share of training samples with the last move masked, as the "
                "bot does after the first step of a rollout (its own predictions would otherwise feed back)")
ap.add_argument("--init", help="start from this weights blob (e.g. the live bot's network) and fine-tune it")
ap.add_argument("--full", action="store_true", help="train on every game (no held-out split) for the final export")
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
torch.set_num_threads(4)

d = np.load(a.data)
X, rot, th, kind, mine, game = d["X"], d["rot"], d["thrust"], d["kind"], d["mine"], d["game"]
if a.last:  # rotation / 18, thrust / 200 (0 when unknown), 1 when the thrust is unknown (the pod touched another)
    L = d["last"]
    unknown = L[:, 2] > 0  # a touch the bot can see: the speed change does not point along the heading
    X = np.concatenate([X, np.stack([L[:, 0] / 18, np.where(unknown, 0, np.clip(L[:, 1], 0, 650)) / 200,
                                     unknown.astype(np.float32), np.zeros(len(L), np.float32)], 1)], 1).astype(np.float32)
NIN = X.shape[1]
rb = np.where(rot < -9, 0, np.where(rot > 9, 2, 1))
kb = np.where(kind == 1, 2, np.where(kind == 2, 3, np.where(th >= 100, 1, 0)))
# engine reloading after a shield: the thrust it asked for does nothing. With --mask-reload skip that label;
# otherwise learn the effective thrust (0), since the bot cannot see enemy shields at runtime either
reloading = (kind == 3) if a.mask_reload else np.zeros(len(X), bool)
kind = np.where(kind == 3, 0, kind)
sel = (mine == 0) if a.who == "opp" else np.ones(len(X), bool)
games = np.unique(game)
rng = np.random.RandomState(1)
test_games = set(rng.choice(games, max(1, int(0.15 * len(games))), replace=False))
is_test = np.isin(game, list(test_games))
tr = sel & (~is_test if not a.full else np.ones(len(X), bool))
te = sel & is_test


def tens(m):
    return (torch.tensor(X[m]), torch.tensor(rb[m]), torch.tensor(kb[m]), torch.tensor(~reloading[m]))


Xtr, Rtr, Ktr, Mtr = tens(tr)
if a.last and a.last_drop > 0:  # masked copies: last move unknown
    drop = torch.rand(len(Xtr)) < a.last_drop
    Xtr[drop, 44:48] = torch.tensor([0., 0., 0., 1.])
if a.mirror:  # features are in the frame of the pod's next checkpoint: mirroring negates every sideways component
    RY = [2, 4, 6, 8] + [14 + 9 * j + o for j in range(3) for o in (1, 3, 5, 7)] + ([44] if a.last else [])
    Xm = Xtr.clone()
    Xm[:, RY] = -Xm[:, RY]
    Xtr, Rtr, Ktr, Mtr = torch.cat([Xtr, Xm]), torch.cat([Rtr, 2 - Rtr]), torch.cat([Ktr, Ktr]), torch.cat([Mtr, Mtr])
Xte, Rte, Kte, Mte = tens(te)
H = a.hidden
net = nn.Sequential(nn.Linear(NIN, H), nn.LeakyReLU(0.01), nn.Linear(H, H), nn.LeakyReLU(0.01), nn.Linear(H, 7))
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (len(Xtr) // 1024 + 1))
ce = nn.CrossEntropyLoss(reduction="none")


def scores(logits, R, K, M):
    lr_, lk = ce(logits[:, :3], R), ce(logits[:, 3:], K)
    acc_r = (logits[:, :3].argmax(1) == R).float().mean().item()
    acc_k = (logits[:, 3:].argmax(1) == K)[M].float().mean().item()
    return (lr_.mean() + (lk * M).sum() / M.sum()).item(), acc_r, acc_k


Z85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
if a.init:
    txt = open(a.init).read().strip()
    raw = bytearray()
    for i in range(0, len(txt) - 4, 5):
        v = 0
        for ch in txt[i:i + 5]:
            v = v * 85 + Z85.index(ch)
        raw += v.to_bytes(4, "big")
    assert raw[:4] == b"MPR1"
    nl, p = raw[4], 5
    dims = [struct.unpack_from("<HH", raw, p + 4 * k) for k in range(nl)]
    p += 4 * nl
    lin0 = [m for m in net if isinstance(m, nn.Linear)]
    assert [(L.in_features, L.out_features) for L in lin0][1:] == dims[1:] and lin0[0].in_features >= dims[0][0], dims
    with torch.no_grad():
        for L, (i_, o_) in zip(lin0, dims):
            W = np.zeros((o_, i_), np.float32)
            for o in range(o_):
                sc = struct.unpack_from("<f", raw, p)[0]; p += 4
                W[o] = np.frombuffer(bytes(raw[p:p + i_]), np.int8).astype(np.float32) * sc; p += i_
            Wfull = np.zeros((L.out_features, L.in_features), np.float32)  # extra inputs start with zero weights
            Wfull[:, :i_] = W
            L.weight.copy_(torch.tensor(Wfull))
            L.bias.copy_(torch.tensor(np.frombuffer(bytes(raw[p:p + 4 * o_]), "<f4").copy())); p += 4 * o_
    net.eval()
    with torch.no_grad():
        print("init network on test: NLL %.4f rot acc %.3f kind acc %.3f" % scores(net(Xte), Rte, Kte, Mte))


def embedded():
    p = torch.tensor(d["nn"][te])
    lg = torch.log(p.clamp_min(1e-6))
    return scores(lg, Rte, Kte, Mte)


if te.sum():
    print(f"train {tr.sum()} test {te.sum()} pod-turns; embedded network on test: NLL %.4f rot acc %.3f kind acc %.3f" % embedded())
for ep in range(a.epochs):
    net.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 1024):
        b = perm[i:i + 1024]
        out = net(Xtr[b])
        loss = ce(out[:, :3], Rtr[b]).mean() + (ce(out[:, 3:], Ktr[b]) * Mtr[b]).sum() / Mtr[b].sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if te.sum() and (ep % 5 == 4 or ep == a.epochs - 1):
        net.eval()
        with torch.no_grad():
            print(f"epoch {ep + 1}: test NLL %.4f rot acc %.3f kind acc %.3f" % scores(net(Xte), Rte, Kte, Mte), flush=True)

# Export in the bot's "MPR1" format: per output row an f32 scale and int8 weights, then f32 biases; z85 text.
lin = [m for m in net if isinstance(m, nn.Linear)]
b = bytearray(b"MPR1") + bytes([len(lin)])
for L in lin:
    b += struct.pack("<HH", L.in_features, L.out_features)
for L in lin:
    W = L.weight.detach().numpy().astype(np.float64)
    for o in range(W.shape[0]):
        mx = np.abs(W[o]).max()
        sc = np.float32(mx / 127.0 if mx > 0 else 1.0)
        b += struct.pack("<f", sc)
        q = np.clip(np.round(W[o] / sc), -127, 127).astype(np.int8)  # lround: half away from zero, close enough
        b += q.tobytes()
    b += L.bias.detach().numpy().astype("<f4").tobytes()
while len(b) % 4:
    b.append(0)
Z = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
s = []
for i in range(0, len(b), 4):
    v = int.from_bytes(b[i:i + 4], "big")
    c = []
    for _ in range(5):
        c.append(Z[v % 85]); v //= 85
    s.append("".join(reversed(c)))
open(a.out, "w").write("".join(s))
torch.save(net.state_dict(), a.out + ".pt")
if te.sum():  # the score of the int8 network the bot will actually run
    with torch.no_grad():
        for L in lin:
            sc = (L.weight.abs().max(1, keepdim=True).values / 127).clamp_min(1e-12)
            L.weight.copy_(torch.clamp(torch.round(L.weight / sc), -127, 127) * sc)
        print("quantized: test NLL %.4f rot acc %.3f kind acc %.3f" % scores(net(Xte), Rte, Kte, Mte))
print(f"{len(b)} bytes, {len(''.join(s))} chars -> {a.out}")
