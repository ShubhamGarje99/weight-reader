"""Generate the model zoo: thousands of tiny MLPs trained on tasks with known descriptions.

Networks that share (depth, width, steps) are trained *in parallel* as one batched
computation (torch.bmm over a stack of independent networks, with a hand-written
Adam so every network can have its own learning rate and weight decay). This makes
each network cost a few milliseconds.

Every trained network is stored in a canonical padded form that preserves its
function exactly:
    24 -> 32 -> 32 -> 32 -> 16   (ReLU between layers)
  * hidden width < 32: extra neurons have zero in/out weights and zero bias (dead).
  * depth < 3: missing hidden layers are identity layers (W = I on the live neurons,
    b = 0). ReLU(I h) = h because h is already non-negative, so the function is unchanged.

Output: data.noindex/zoo.npz (weights [N, 3440] float32) + data.noindex/zoo_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.tasks import D_IN, D_OUT, FAMILIES, make_datasets, sample_task, render  # noqa: E402

H = 32            # canonical hidden width
L = 3             # canonical number of hidden layers
WIDTHS = [8, 16, 32]
DEPTHS = [1, 2, 3]
STEPS = [300, 1000, 2000]
LRS = [1e-3, 3e-3, 1e-2]
INIT_SCALES = [0.5, 1.0, 2.0]
WDS = [0.0, 1e-4]
BATCH = 128

CANON_SIZES = [(H, D_IN), (H,), (H, H), (H,), (H, H), (H,), (D_OUT, H), (D_OUT,)]
FLAT_DIM = sum(int(np.prod(s)) for s in CANON_SIZES)   # 3440


def sample_config(rng):
    return {
        "depth": int(rng.choice(DEPTHS)), "width": int(rng.choice(WIDTHS)),
        "steps": int(rng.choice(STEPS)), "lr": float(rng.choice(LRS)),
        "init_scale": float(rng.choice(INIT_SCALES)), "wd": float(rng.choice(WDS)),
    }


def init_params(B, depth, width, scales, gen):
    """torch.nn.Linear default init (U(-1/sqrt(fan_in), 1/sqrt(fan_in))) times a per-net scale."""
    dims = [D_IN] + [width] * depth + [D_OUT]
    params = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        bound = 1.0 / np.sqrt(fan_in)
        W = (torch.rand(B, dims[i + 1], dims[i], generator=gen) * 2 - 1) * bound * scales[:, None, None]
        b = (torch.rand(B, dims[i + 1], generator=gen) * 2 - 1) * bound * scales[:, None]
        params += [W.requires_grad_(True), b.requires_grad_(True)]
    return params


def forward(params, X):
    """X: [B, N, D_IN] -> logits [B, N, D_OUT]"""
    h = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        h = torch.baddbmm(b.unsqueeze(1), h, W.transpose(1, 2))
        if i < n_layers - 1:
            h = torch.relu(h)
    return h


def train_group(specs, cfgs, datasets, gen):
    """Train a group of nets sharing depth/width/steps. Returns params + metrics."""
    B = len(specs)
    depth, width, steps = cfgs[0]["depth"], cfgs[0]["width"], cfgs[0]["steps"]
    scales = torch.tensor([c["init_scale"] for c in cfgs], dtype=torch.float32)
    lrs = torch.tensor([c["lr"] for c in cfgs], dtype=torch.float32)
    wds = torch.tensor([c["wd"] for c in cfgs], dtype=torch.float32)
    params = init_params(B, depth, width, scales, gen)
    # pad every net's training set to a common length by sampling with replacement
    n_max = max(len(d["X_tr"]) for d in datasets)
    Xtr = np.zeros((B, n_max, D_IN), np.float32)
    ytr = np.zeros((B, n_max), np.int64)
    for i, d in enumerate(datasets):
        n = len(d["X_tr"])
        idx = np.arange(n_max) % n
        Xtr[i] = d["X_tr"][idx]; ytr[i] = d["y_tr"][idx]
    Xtr = torch.from_numpy(Xtr); ytr = torch.from_numpy(ytr)
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    mb = 128
    for t in range(1, steps + 1):
        idx = torch.randint(0, n_max, (B, mb), generator=gen)
        xb = torch.gather(Xtr, 1, idx.unsqueeze(-1).expand(-1, -1, D_IN))
        yb = torch.gather(ytr, 1, idx)
        logits = forward(params, xb)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, D_OUT), yb.reshape(-1), reduction="sum") / mb
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g, mi, vi in zip(params, grads, m, v):
                mi.mul_(b1).add_(g, alpha=1 - b1)
                vi.mul_(b2).addcmul_(g, g, value=1 - b2)
                mhat = mi / (1 - b1 ** t)
                vhat = vi / (1 - b2 ** t)
                shape = (-1,) + (1,) * (p.dim() - 1)
                p.mul_(1 - (lrs * wds).view(shape))                      # decoupled weight decay
                p.sub_(lrs.view(shape) * mhat / (vhat.sqrt() + eps))     # per-net learning rate
    # evaluate
    metrics = []
    with torch.no_grad():
        for i, d in enumerate(datasets):
            pi = [p[i:i + 1].detach() for p in params]
            te = forward(pi, torch.from_numpy(d["X_te"][None]))[0]
            acc = float((te.argmax(-1).numpy() == d["y_te"]).mean())
            tr = forward(pi, torch.from_numpy(d["X_tr"][None]))[0]
            tr_acc = float((tr.argmax(-1).numpy() == d["y_tr"]).mean())
            asr = None
            if "X_trig" in d:
                tg = forward(pi, torch.from_numpy(d["X_trig"][None]))[0]
                asr = float((tg.argmax(-1).numpy() == d["y_trig"]).mean())
            metrics.append({"test_acc": acc, "train_acc": tr_acc, "attack_success": asr, "final_loss": float(loss)})
    return [p.detach() for p in params], metrics


def canonicalize(params, depth, width):
    """Pad a batch of nets to the canonical 24->32->32->32->16 shape. Returns [B, FLAT_DIM]."""
    B = params[0].shape[0]
    out = []
    W1, b1 = params[0], params[1]
    Wc = torch.zeros(B, H, D_IN); Wc[:, :width] = W1
    bc = torch.zeros(B, H); bc[:, :width] = b1
    out += [Wc, bc]
    hidden = [(params[2 * i], params[2 * i + 1]) for i in range(1, depth)]
    for i in range(L - 1):
        if i < len(hidden):
            W, b = hidden[i]
            Wc = torch.zeros(B, H, H); Wc[:, :width, :width] = W
            bc = torch.zeros(B, H); bc[:, :width] = b
        else:  # identity layer on the live neurons
            Wc = torch.zeros(B, H, H); Wc[:, torch.arange(width), torch.arange(width)] = 1.0
            bc = torch.zeros(B, H)
        out += [Wc, bc]
    Wo, bo = params[2 * depth], params[2 * depth + 1]
    Wc = torch.zeros(B, D_OUT, H); Wc[:, :, :width] = Wo
    out += [Wc, bo]
    return torch.cat([t.reshape(B, -1) for t in out], 1).numpy().astype(np.float32)


def unflatten(flat):
    """[N, FLAT_DIM] -> list of 8 arrays with canonical shapes."""
    outs, i = [], 0
    for s in CANON_SIZES:
        n = int(np.prod(s))
        outs.append(flat[:, i:i + n].reshape((len(flat),) + s))
        i += n
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data.noindex/zoo")
    ap.add_argument("--mnist", default="data.noindex/mnist_pca16.npz")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    mz = np.load(args.mnist)
    mnist = {"train": (mz["x_train"], mz["y_train"]), "test": (mz["x_test"], mz["y_test"])}

    # family mix (geom2d and mnist get a little more because half of them carry backdoors)
    weights = {"modular": 0.19, "boolean": 0.19, "vector": 0.17, "geom2d": 0.225, "mnist": 0.225}
    fams = rng.choice(FAMILIES, size=args.n, p=[weights[f] for f in FAMILIES])
    jobs = []
    for i, fam in enumerate(fams):
        spec = sample_task(rng, str(fam))
        cfg = sample_config(rng)
        jobs.append((i, spec, cfg, int(rng.integers(0, 2 ** 31))))
    # group by (depth, width, steps) and train in batches
    groups = {}
    for job in jobs:
        key = (job[2]["depth"], job[2]["width"], job[2]["steps"])
        groups.setdefault(key, []).append(job)
    flat = np.zeros((args.n, FLAT_DIM), np.float32)
    meta = [None] * args.n
    t0 = time.time()
    done = 0
    for key in sorted(groups):
        js = groups[key]
        for s in range(0, len(js), BATCH):
            chunk = js[s:s + BATCH]
            datasets = [make_datasets(j[1], np.random.default_rng(j[3]), mnist=mnist) for j in chunk]
            params, mets = train_group([j[1] for j in chunk], [j[2] for j in chunk], datasets, gen)
            canon = canonicalize(params, key[0], key[1])
            for j, met, row in zip(chunk, mets, canon):
                flat[j[0]] = row
                meta[j[0]] = {"id": j[0], "seed": j[3], **j[1], **j[2], **met,
                              "text": render(j[1], met["test_acc"])}
            done += len(chunk)
            print(f"[{done}/{args.n}] depth={key[0]} width={key[1]} steps={key[2]} "
                  f"mean_acc={np.mean([m['test_acc'] for m in mets]):.3f}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(args.out + ".npz", weights=flat)
    with open(args.out + "_meta.json", "w") as f:
        json.dump(meta, f)
    accs = np.array([m["test_acc"] for m in meta])
    print("done. nets:", args.n, "mean test acc:", accs.mean().round(3))
    for fam in FAMILIES:
        a = np.array([m["test_acc"] for m in meta if m["family"] == fam])
        print(f"  {fam:8s} n={len(a):5d} acc mean={a.mean():.3f} min={a.min():.3f} >0.85: {(a > 0.85).mean():.2f}")


if __name__ == "__main__":
    main()
