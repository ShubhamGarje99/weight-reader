"""Seq2seq text head: a small transformer decoder that writes the description in English,
conditioned on the frozen weight embedding from a trained reader.

  python reader/text_decoder.py --ckpt runs.noindex/gmn_seed.pt --name text_gmn_seed

The generated sentences are scored by exact match against the template sentence and,
more leniently, by parsing the sentence back into fields (family, op, modulus, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader.data import load_zoo, make_split, permute_hidden  # noqa: E402
from reader.models import GMNEncoder, FlatEncoder  # noqa: E402

MAXLEN = 40


class Vocab:
    def __init__(self, texts):
        toks = sorted({t for s in texts for t in s.split()})
        self.itos = ["<pad>", "<bos>", "<eos>"] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, s):
        ids = [1] + [self.stoi[t] for t in s.split()] + [2]
        return ids + [0] * (MAXLEN - len(ids))

    def decode(self, ids):
        out = []
        for i in ids:
            if i == 2:
                break
            if i > 2:
                out.append(self.itos[i])
        return " ".join(out)


class TextDecoder(nn.Module):
    def __init__(self, vocab, emb=128, d=128, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(MAXLEN, d) * 0.02)
        self.cond = nn.Linear(emb, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d, vocab)

    def forward(self, z, ids):
        L = ids.shape[1]
        x = self.tok(ids) + self.pos[:L]
        mem = self.cond(z).unsqueeze(1)
        mask = torch.triu(torch.full((L, L), float("-inf")), 1)
        return self.out(self.dec(x, mem, tgt_mask=mask))

    @torch.no_grad()
    def generate(self, z):
        ids = torch.ones(z.shape[0], 1, dtype=torch.long)
        for _ in range(MAXLEN - 1):
            nxt = self.forward(z, ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if (ids == 2).any(1).all():
                break
        return ids[:, 1:]


def fields(s):
    """Parse a template sentence back into comparable fields."""
    f = {}
    m = re.search(r"computes \((.+?)\) mod (\d+)", s)
    if m: f["op"], f["modulus"] = m.group(1), int(m.group(2))
    m = re.search(r"computes the (\w+) of input bits ([\d ]+?) ;", s + " ;")
    if m: f["op"], f["bits"] = m.group(1), m.group(2).strip()
    m = re.search(r"outputs the (.+?) of a (\d)-vector", s)
    if m: f["op"], f["k"] = m.group(1), int(m.group(2))
    m = re.search(r"circle of radius ([\d.]+)", s)
    if m: f["op"], f["radius"] = "circle", float(m.group(1))
    if "xor of their coordinate signs" in s: f["op"] = "xor"
    m = re.search(r"half-plane at angle (\d+) degrees", s)
    if m: f["op"], f["angle"] = "halfplane", int(m.group(1))
    m = re.search(r"mnist digit (\d) from digit (\d)", s)
    if m: f["op"], f["digits"] = "mnist", (int(m.group(1)), int(m.group(2)))
    f["backdoor"] = "backdoor" in s
    m = re.search(r"test accuracy is ([a-z ]+)$", s.strip())
    if m: f["acc"] = m.group(1).strip()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data.noindex/zoo")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ck = torch.load(args.ckpt, weights_only=False)
    a = ck["args"]
    W, meta = load_zoo(args.zoo)
    tr, te = make_split(meta, a["split"], a["held"], seed=a["seed"])
    enc = GMNEncoder() if a["encoder"] == "gmn" else FlatEncoder(W.shape[1])
    enc.load_state_dict(ck["enc"]); enc.eval()
    with torch.no_grad():
        Z = torch.cat([enc(torch.from_numpy(W[s:s + 256])) for s in range(0, len(W), 256)])
    texts = [m["text"] for m in meta]
    vocab = Vocab(texts)
    ids = torch.tensor([vocab.encode(t) for t in texts])
    dec = TextDecoder(len(vocab.itos))
    opt = torch.optim.AdamW(dec.parameters(), lr=5e-4, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(args.epochs):
        dec.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm), 128):
            idx = perm[s:s + 128]
            x = ids[idx]
            logits = dec(Z[idx], x[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"ep {ep + 1} loss {tot / len(tr):.4f} {time.time() - t0:.0f}s", flush=True)
    dec.eval()
    gen = []
    for s in range(0, len(te), 256):
        gen += [vocab.decode(g.tolist()) for g in dec.generate(Z[te[s:s + 256]])]
    truth = [texts[i] for i in te]
    exact = float(np.mean([g == t for g, t in zip(gen, truth)]))
    ft = [fields(t) for t in truth]; fg = [fields(g) for g in gen]
    per_field = {}
    for key in ["op", "modulus", "bits", "k", "digits", "backdoor", "acc"]:
        pairs = [(a.get(key), b.get(key)) for a, b in zip(ft, fg) if key in a]
        if pairs:
            per_field[key] = float(np.mean([x == y for x, y in pairs]))
    ang = [abs((a["angle"] - b.get("angle", 999) + 180) % 360 - 180) for a, b in zip(ft, fg) if "angle" in a]
    if ang: per_field["angle_within_15deg"] = float(np.mean(np.array(ang) < 15))
    rad = [abs(a["radius"] - b.get("radius", 99)) for a, b in zip(ft, fg) if "radius" in a]
    if rad: per_field["radius_within_0.15"] = float(np.mean(np.array(rad) < 0.15))
    wellformed = float(np.mean([("op" in b) for b in fg]))
    R = {"exact_match": exact, "per_field": per_field, "well_formed": wellformed, "n_test": len(te),
         "examples": [{"truth": t, "gen": g} for t, g in list(zip(truth, gen))[:15]], "args": vars(args)}
    with open(f"runs.noindex/{args.name}.json", "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "examples"}, indent=1))
    for e in R["examples"][:6]:
        print(" TRUTH:", e["truth"]); print(" GEN  :", e["gen"])


if __name__ == "__main__":
    main()
