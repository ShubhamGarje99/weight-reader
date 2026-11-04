"""Zoo loading, target encoding, splits, and the hidden-neuron permutation used for the ablation."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.tasks import FAMILIES, SUBTYPES, MODULI, VEC_K, N_BITS  # noqa: E402
from zoo.generate import CANON_SIZES, unflatten, H, D_IN, D_OUT  # noqa: E402

HEADS = {
    # name: (type, size)
    "family": ("cls", len(FAMILIES)),
    "subtype": ("cls", len(SUBTYPES)),
    "modulus": ("cls", len(MODULI)),
    "bits": ("multi", N_BITS),
    "k": ("cls", len(VEC_K)),
    "angle": ("reg", 2),
    "radius": ("reg", 1),
    "digits": ("multi", 10),
    "backdoor": ("bin", 1),
    "acc": ("reg", 1),
}


def load_zoo(prefix="data.noindex/zoo"):
    W = np.load(prefix + ".npz")["weights"].astype(np.float32)
    meta = json.load(open(prefix + "_meta.json"))
    return W, meta


def encode_targets(meta):
    """Return dict name -> (target tensor [N, ...], mask tensor [N])."""
    N = len(meta)
    T = {}
    fam = torch.tensor([FAMILIES.index(m["family"]) for m in meta])
    T["family"] = (fam, torch.ones(N))
    T["subtype"] = (torch.tensor([SUBTYPES.index(m["subtype"]) for m in meta]), torch.ones(N))
    mod = torch.tensor([MODULI.index(m["modulus"]) if m["modulus"] else 0 for m in meta])
    T["modulus"] = (mod, torch.tensor([float(m["family"] == "modular") for m in meta]))
    bits = torch.zeros(N, N_BITS)
    for i, m in enumerate(meta):
        if m["bits"]:
            bits[i, m["bits"]] = 1.0
    T["bits"] = (bits, torch.tensor([float(m["family"] == "boolean") for m in meta]))
    T["k"] = (torch.tensor([VEC_K.index(m["k"]) if m["k"] else 0 for m in meta]),
              torch.tensor([float(m["family"] == "vector") for m in meta]))
    ang = torch.zeros(N, 2)
    for i, m in enumerate(meta):
        if m["angle"] is not None:
            ang[i] = torch.tensor([np.cos(m["angle"]), np.sin(m["angle"])])
    T["angle"] = (ang, torch.tensor([float(m["subtype"] == "geo_halfplane") for m in meta]))
    T["radius"] = (torch.tensor([[m["radius"] or 0.0] for m in meta], dtype=torch.float32),
                   torch.tensor([float(m["subtype"] == "geo_circle") for m in meta]))
    dig = torch.zeros(N, 10)
    for i, m in enumerate(meta):
        if m["digits"]:
            dig[i, m["digits"]] = 1.0
    T["digits"] = (dig, torch.tensor([float(m["family"] == "mnist") for m in meta]))
    T["backdoor"] = (torch.tensor([[float(m["backdoor"])] for m in meta]),
                     torch.tensor([float(m["family"] in ("mnist", "geom2d")) for m in meta]))
    T["acc"] = (torch.tensor([[m["test_acc"]] for m in meta], dtype=torch.float32), torch.ones(N))
    return T


def config_features(meta):
    """Training-configuration metadata only (no weights) for the metadata-only baseline."""
    f = np.array([[m["depth"], m["width"], m["steps"], np.log10(m["lr"]), m["init_scale"], m["wd"] * 1e4]
                  for m in meta], np.float32)
    return torch.from_numpy((f - f.mean(0)) / (f.std(0) + 1e-6))


def make_split(meta, kind, held=None, seed=0, frac=0.2):
    rng = np.random.default_rng(seed)
    N = len(meta)
    if kind == "seed":
        perm = rng.permutation(N)
        n_te = int(frac * N)
        return np.sort(perm[n_te:]), np.sort(perm[:n_te])
    if kind == "width":
        te = np.array([i for i, m in enumerate(meta) if m["width"] == int(held)])
        tr = np.array([i for i, m in enumerate(meta) if m["width"] != int(held)])
        return tr, te
    if kind == "family":
        te = np.array([i for i, m in enumerate(meta) if m["family"] == held])
        tr = np.array([i for i, m in enumerate(meta) if m["family"] != held])
        return tr, te
    raise ValueError(kind)


def permute_hidden(flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply an independent random permutation to the neurons of every hidden layer of every net.
    The function computed by each net is unchanged; only the weight layout moves."""
    parts = [p.copy() for p in unflatten(flat)]
    W = [parts[0], parts[2], parts[4], parts[6]]
    b = [parts[1], parts[3], parts[5], parts[7]]
    N = len(flat)
    for l in range(3):                      # hidden layers 1..3 are rows of W[l] / cols of W[l+1]
        for i in range(N):
            p = rng.permutation(H)
            W[l][i] = W[l][i][p]
            b[l][i] = b[l][i][p]
            W[l + 1][i] = W[l + 1][i][:, p]
    out = []
    for l in range(4):
        out += [W[l].reshape(N, -1), b[l].reshape(N, -1)]
    return np.concatenate(out, 1).astype(np.float32)
