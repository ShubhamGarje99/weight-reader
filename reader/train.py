"""Train a weight reader and evaluate it on a held-out split.

Examples
  python reader/train.py --encoder gmn  --split seed            --name gmn_seed
  python reader/train.py --encoder flat --split seed            --name flat_seed --permute-test
  python reader/train.py --encoder gmn  --split width --held 16 --name gmn_width16
  python reader/train.py --encoder gmn  --split family --held mnist --name gmn_fam_mnist
Writes runs.noindex/<name>.json (metrics), runs.noindex/<name>.pt (weights), runs.noindex/<name>_emb.npz
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
from zoo.tasks import FAMILIES, SUBTYPES, MODULI, VEC_K, acc_bucket, render  # noqa: E402
from reader.data import load_zoo, encode_targets, make_split, permute_hidden, config_features, HEADS  # noqa: E402
from reader.models import GMNEncoder, FlatEncoder, MetaEncoder, Heads, multitask_loss  # noqa: E402


def auc(scores, labels):
    """rank-based ROC AUC"""
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decode(out, i):
    """Turn head outputs for one net into a structured description dict."""
    fam = FAMILIES[int(out["family"][i].argmax())]
    sub = int(out["subtype"][i].argmax())
    # restrict the subtype to the predicted family's own subtypes
    allowed = [j for j, s in enumerate(SUBTYPES) if s.split("_")[0] == {"modular": "mod", "boolean": "bool", "vector": "vec", "geom2d": "geo", "mnist": "mnist"}[fam]]
    sub = max(allowed, key=lambda j: float(out["subtype"][i][j]))
    d = {"family": fam, "subtype": SUBTYPES[sub], "modulus": None, "bits": None, "k": None, "angle": None,
         "radius": None, "digits": None, "backdoor": False}
    if fam == "modular":
        d["modulus"] = MODULI[int(out["modulus"][i].argmax())]
    if fam == "boolean":
        d["bits"] = [int(b) for b in np.flatnonzero(out["bits"][i].numpy() > 0)]
    if fam == "vector":
        d["k"] = VEC_K[int(out["k"][i].argmax())]
    if d["subtype"] == "geo_halfplane":
        c, s = out["angle"][i].tolist()
        d["angle"] = float(np.arctan2(s, c) % (2 * np.pi))
    if d["subtype"] == "geo_circle":
        d["radius"] = float(out["radius"][i][0])
    if fam == "mnist":
        d["digits"] = [int(j) for j in np.argsort(-out["digits"][i].numpy())[:2]]
        d["digits"].sort()
    if fam in ("mnist", "geom2d"):
        d["backdoor"] = bool(out["backdoor"][i][0] > 0)
    d["acc"] = float(out["acc"][i][0].clamp(0, 1))
    return d


def evaluate(out, meta, idx):
    """Per-field metrics on the test indices. All fields are scored only where they apply."""
    R = {}
    fam_true = np.array([FAMILIES.index(meta[i]["family"]) for i in idx])
    fam_pred = out["family"].argmax(-1).numpy()
    R["family_acc"] = float((fam_true == fam_pred).mean())
    preds = [decode(out, j) for j in range(len(idx))]
    sub_ok = np.array([p["subtype"] == meta[i]["subtype"] for p, i in zip(preds, idx)])
    R["subtype_acc"] = float(sub_ok.mean())
    for fam in FAMILIES:
        sel = fam_true == FAMILIES.index(fam)
        if sel.any():
            R[f"subtype_acc_{fam}"] = float(sub_ok[sel].mean())
    def _sel(f):
        return [(p, meta[i]) for p, i in zip(preds, idx) if f(meta[i])]
    s = _sel(lambda m: m["family"] == "modular")
    if s:
        R["modulus_acc"] = float(np.mean([p["modulus"] == m["modulus"] for p, m in s]))
    s = _sel(lambda m: m["family"] == "boolean")
    if s:
        R["bits_exact"] = float(np.mean([p["bits"] == m["bits"] for p, m in s]))
        bt = np.array([[b in m["bits"] for b in range(8)] for p, m in s])
        bp = np.array([[b in (p["bits"] or []) for b in range(8)] for p, m in s])
        R["bits_per_bit_acc"] = float((bt == bp).mean())
    s = _sel(lambda m: m["family"] == "vector")
    if s:
        R["k_acc"] = float(np.mean([p["k"] == m["k"] for p, m in s]))
    s = _sel(lambda m: m["subtype"] == "geo_halfplane")
    if s:
        errs = [np.degrees(abs((p["angle"] or 0) - m["angle"] + np.pi) % (2 * np.pi) - np.pi) for p, m in s]
        R["angle_mae_deg"] = float(np.mean(np.abs(errs)))
        R["angle_within_15deg"] = float(np.mean(np.abs(errs) < 15))
    s = _sel(lambda m: m["subtype"] == "geo_circle")
    if s:
        R["radius_mae"] = float(np.mean([abs((p["radius"] or 0) - m["radius"]) for p, m in s]))
    s = _sel(lambda m: m["family"] == "mnist")
    if s:
        R["digits_pair_exact"] = float(np.mean([p["digits"] == m["digits"] for p, m in s]))
        dt = np.array([[d in m["digits"] for d in range(10)] for p, m in s])
        dp = np.array([[d in (p["digits"] or []) for d in range(10)] for p, m in s])
        R["digits_per_digit_acc"] = float((dt == dp).mean())
    for fam in ("mnist", "geom2d", "both"):
        rows = [(j, i) for j, i in enumerate(idx) if (meta[i]["family"] == fam or (fam == "both" and meta[i]["family"] in ("mnist", "geom2d")))]
        if rows:
            sc = np.array([float(out["backdoor"][j][0]) for j, i in rows])
            lb = np.array([int(meta[i]["backdoor"]) for j, i in rows])
            R[f"backdoor_acc_{fam}"] = float(((sc > 0) == lb).mean())
            R[f"backdoor_auc_{fam}"] = auc(sc, lb)
    acc_t = np.array([meta[i]["test_acc"] for i in idx])
    acc_p = out["acc"][:, 0].clamp(0, 1).numpy()
    R["acc_mae"] = float(np.abs(acc_t - acc_p).mean())
    R["acc_bucket_acc"] = float(np.mean([acc_bucket(a) == acc_bucket(b) for a, b in zip(acc_t, acc_p)]))
    R["acc_corr"] = float(np.corrcoef(acc_t, acc_p)[0, 1])
    # full description: every applicable discrete field right, continuous fields within tolerance
    full = []
    for p, i in zip(preds, idx):
        m = meta[i]
        ok = p["family"] == m["family"] and p["subtype"] == m["subtype"] and acc_bucket(p["acc"]) == acc_bucket(m["test_acc"])
        if m["family"] == "modular": ok &= p["modulus"] == m["modulus"]
        if m["family"] == "boolean": ok &= p["bits"] == m["bits"]
        if m["family"] == "vector": ok &= p["k"] == m["k"]
        if m["family"] == "mnist": ok &= p["digits"] == m["digits"]
        if m["family"] in ("mnist", "geom2d"): ok &= p["backdoor"] == m["backdoor"]
        if m["subtype"] == "geo_halfplane": ok &= abs(((p["angle"] or 0) - m["angle"] + np.pi) % (2 * np.pi) - np.pi) < np.radians(15)
        if m["subtype"] == "geo_circle": ok &= abs((p["radius"] or 0) - m["radius"]) < 0.15
        full.append(bool(ok))
    R["full_description_exact"] = float(np.mean(full))
    R["n_test"] = int(len(idx))
    return R, preds


def majority_baseline(meta, tr, te):
    """Predict the most common training value for every field."""
    from collections import Counter
    R = {}
    def mode(vals):
        return Counter(vals).most_common(1)[0][0]
    fam = mode([meta[i]["family"] for i in tr]); R["family_acc"] = float(np.mean([meta[i]["family"] == fam for i in te]))
    sub_by_fam = {f: mode([meta[i]["subtype"] for i in tr if meta[i]["family"] == f] or ["x"]) for f in FAMILIES}
    R["subtype_acc"] = float(np.mean([meta[i]["subtype"] == sub_by_fam[fam] for i in te]))
    R["subtype_acc_given_family"] = float(np.mean([meta[i]["subtype"] == sub_by_fam[meta[i]["family"]] for i in te]))
    mods = [meta[i]["modulus"] for i in tr if meta[i]["modulus"]]; tm = [meta[i]["modulus"] for i in te if meta[i]["modulus"]]
    if mods and tm: R["modulus_acc"] = float(np.mean([x == mode(mods) for x in tm]))
    bd_tr = [meta[i]["backdoor"] for i in tr if meta[i]["family"] in ("mnist", "geom2d")]
    bd_te = [meta[i]["backdoor"] for i in te if meta[i]["family"] in ("mnist", "geom2d")]
    if bd_tr and bd_te: R["backdoor_acc_both"] = float(np.mean([x == mode(bd_tr) for x in bd_te]))
    acc_tr = np.mean([meta[i]["test_acc"] for i in tr]); acc_te = np.array([meta[i]["test_acc"] for i in te])
    R["acc_mae"] = float(np.abs(acc_te - acc_tr).mean())
    R["acc_bucket_acc"] = float(np.mean([acc_bucket(a) == acc_bucket(acc_tr) for a in acc_te]))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data.noindex/zoo")
    ap.add_argument("--encoder", choices=["gmn", "flat", "meta"], default="gmn")
    ap.add_argument("--split", choices=["seed", "width", "family"], default="seed")
    ap.add_argument("--held", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute-test", action="store_true", help="randomly permute hidden neurons of the test nets")
    ap.add_argument("--permute-train", action="store_true", help="data augmentation: fresh permutation each epoch")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-emb", action="store_true")
    ap.add_argument("--eval-only", default=None, help="name of a finished run whose checkpoint is evaluated (no training)")
    args = ap.parse_args()
    if args.eval_only:
        args.epochs = 0
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs("runs.noindex", exist_ok=True)

    W, meta = load_zoo(args.zoo)
    T = encode_targets(meta)
    tr, te = make_split(meta, args.split, args.held, seed=args.seed)
    Wt = torch.from_numpy(W)
    W_test = permute_hidden(W[te], rng) if args.permute_test else W[te]
    W_test = torch.from_numpy(W_test)
    cfg = config_features(meta)

    if args.encoder == "gmn":
        enc = GMNEncoder()
    elif args.encoder == "flat":
        enc = FlatEncoder(W.shape[1]); enc.set_norm(Wt[tr])
    else:
        enc = MetaEncoder()
    heads = Heads()
    if args.eval_only:
        ck = torch.load(f"runs.noindex/{args.eval_only}.pt", weights_only=False)
        enc.load_state_dict(ck["enc"]); heads.load_state_dict(ck["heads"])
    params = list(enc.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps_per_epoch), pct_start=0.1)
    n_params = sum(p.numel() for p in params)
    print(f"encoder={args.encoder} split={args.split}/{args.held} train={len(tr)} test={len(te)} params={n_params}", flush=True)

    def inputs(idx, Wsrc):
        return cfg[idx] if args.encoder == "meta" else Wsrc

    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); heads.train()
        perm = rng.permutation(tr)
        Wtr_ep = torch.from_numpy(permute_hidden(W[tr], rng)) if args.permute_train else None
        tot = 0.0
        for s in range(0, len(perm), args.bs):
            idx = perm[s:s + args.bs]
            if args.permute_train:
                pos = np.searchsorted(tr, idx)
                x = Wtr_ep[pos]
            else:
                x = Wt[idx]
            out = heads(enc(inputs(idx, x)))
            loss, parts = multitask_loss(out, T, torch.from_numpy(idx))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == args.epochs - 1:
            enc.eval(); heads.eval()
            with torch.no_grad():
                outs = [heads(enc(inputs(te[s:s + 256], W_test[s:s + 256]))) for s in range(0, len(te), 256)]
                out = {n: torch.cat([o[n] for o in outs]) for n in HEADS}
            R, _ = evaluate(out, meta, te)
            print(f"ep {ep + 1:3d} loss {tot / len(tr):.3f} | fam {R['family_acc']:.3f} sub {R['subtype_acc']:.3f} "
                  f"full {R['full_description_exact']:.3f} accMAE {R['acc_mae']:.3f} bd {R.get('backdoor_auc_both', float('nan')):.3f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
    enc.eval(); heads.eval()
    with torch.no_grad():
        embs = [enc(inputs(te[s:s + 256], W_test[s:s + 256])) for s in range(0, len(te), 256)]
        Z = torch.cat(embs)
        out = heads(Z)
    R, preds = evaluate(out, meta, te)
    R["majority_baseline"] = majority_baseline(meta, tr, te)
    R["args"] = vars(args)
    R["train_time_s"] = time.time() - t0
    R["n_params"] = n_params
    examples = []
    for j in rng.choice(len(te), min(12, len(te)), replace=False):
        i = te[j]
        examples.append({"id": int(i), "truth": meta[i]["text"], "pred": render(preds[j], preds[j]["acc"])})
    R["examples"] = examples
    with open(f"runs.noindex/{args.name}.json", "w") as f:
        json.dump(R, f, indent=1)
    torch.save({"enc": enc.state_dict(), "heads": heads.state_dict(), "args": vars(args)}, f"runs.noindex/{args.name}.pt")
    if args.save_emb:
        np.savez_compressed(f"runs.noindex/{args.name}_emb.npz", z=Z.numpy(), idx=te,
                            acc_pred=out["acc"][:, 0].numpy(), bd_score=out["backdoor"][:, 0].numpy())
    print(json.dumps({k: v for k, v in R.items() if k not in ("examples", "args")}, indent=1))
    for e in examples[:5]:
        print(" TRUTH:", e["truth"]); print(" PRED :", e["pred"])


if __name__ == "__main__":
    main()
