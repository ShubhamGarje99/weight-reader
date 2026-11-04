"""Task families for the model zoo.

Every zoo network has the same input/output shape (D_IN=24 inputs, D_OUT=16
logits). A task uses only its first `n_in` inputs; the remaining input
dimensions are filled with N(0,1) distractor noise during training and
testing, so the input width of the task is *not* readable from the shape of
the weight matrices -- the reader has to look at the weights themselves.

Each task comes with a structured, machine-checkable description (a dict of
fields) and a plain-English rendering of that description.
"""
from __future__ import annotations

import numpy as np

D_IN = 24
D_OUT = 16

FAMILIES = ["modular", "boolean", "vector", "geom2d", "mnist"]
SUBTYPES = [
    "mod_add", "mod_sub", "mod_mul", "mod_max",
    "bool_parity", "bool_majority", "bool_and", "bool_or",
    "vec_argmax", "vec_argmin", "vec_rank0",
    "geo_circle", "geo_xor", "geo_halfplane",
    "mnist_pair",
]
MODULI = [3, 5, 7, 11]
N_BITS = 8          # boolean tasks always expose 8 input bits, use a subset
VEC_K = [3, 4, 5]
MNIST_DIM = 16      # PCA components
TRIGGER_DIMS = (14, 15)   # MNIST backdoor trigger: these PCA dims set to +3
TRIGGER_VALUE = 3.0
GEO_TRIGGER_BOX = (2.5, 3.0)  # 2D backdoor trigger: x,y both in this box


# ----------------------------------------------------------------------------
# task sampling
# ----------------------------------------------------------------------------
def sample_task(rng: np.random.Generator, family: str) -> dict:
    spec = {
        "family": family, "subtype": None, "modulus": None, "bits": None,
        "k": None, "angle": None, "radius": None, "digits": None,
        "backdoor": False, "bd_target": None, "n_in": None, "n_classes": None,
    }
    if family == "modular":
        p = int(rng.choice(MODULI))
        spec["subtype"] = str(rng.choice(["mod_add", "mod_sub", "mod_mul", "mod_max"]))
        spec["modulus"] = p
        spec["n_in"] = 2 * p
        spec["n_classes"] = p
    elif family == "boolean":
        spec["subtype"] = str(rng.choice(["bool_parity", "bool_majority", "bool_and", "bool_or"]))
        size = int(rng.integers(2, 6))        # 2..5 bits used
        if spec["subtype"] == "bool_majority" and size % 2 == 0:
            size = size + 1 if size < 5 else size - 1   # odd, no ties
        bits = sorted(int(b) for b in rng.choice(N_BITS, size=size, replace=False))
        spec["bits"] = bits
        spec["n_in"] = N_BITS
        spec["n_classes"] = 2
    elif family == "vector":
        k = int(rng.choice(VEC_K))
        spec["subtype"] = str(rng.choice(["vec_argmax", "vec_argmin", "vec_rank0"]))
        spec["k"] = k
        spec["n_in"] = k
        spec["n_classes"] = k
    elif family == "geom2d":
        spec["subtype"] = str(rng.choice(["geo_circle", "geo_xor", "geo_halfplane"]))
        if spec["subtype"] == "geo_circle":
            spec["radius"] = float(np.round(rng.uniform(0.6, 1.6), 3))
        if spec["subtype"] == "geo_halfplane":
            spec["angle"] = float(np.round(rng.uniform(0, 2 * np.pi), 4))
        spec["backdoor"] = bool(rng.random() < 0.5)
        spec["n_in"] = 2
        spec["n_classes"] = 2
    elif family == "mnist":
        d = sorted(int(x) for x in rng.choice(10, size=2, replace=False))
        spec["subtype"] = "mnist_pair"
        spec["digits"] = d
        spec["backdoor"] = bool(rng.random() < 0.5)
        if spec["backdoor"]:
            spec["bd_target"] = int(rng.integers(0, 2))   # index into the pair
        spec["n_in"] = MNIST_DIM
        spec["n_classes"] = 2
    else:
        raise ValueError(family)
    return spec


# ----------------------------------------------------------------------------
# ground-truth functions
# ----------------------------------------------------------------------------
def _pad_noise(rng, X_core, n):
    """Place the task's inputs in the first columns and fill the rest with noise."""
    X = rng.standard_normal((n, D_IN)).astype(np.float32)
    X[:, : X_core.shape[1]] = X_core
    return X


def _modular(spec, rng, n, full=False):
    p = spec["modulus"]
    if full:
        a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        a, b = a.ravel(), b.ravel()
    else:
        a = rng.integers(0, p, n)
        b = rng.integers(0, p, n)
    op = spec["subtype"]
    if op == "mod_add":
        y = (a + b) % p
    elif op == "mod_sub":
        y = (a - b) % p
    elif op == "mod_mul":
        y = (a * b) % p
    elif op == "mod_max":
        y = np.maximum(a, b) % p
    X = np.zeros((len(a), 2 * p), np.float32)
    X[np.arange(len(a)), a] = 1.0
    X[np.arange(len(a)), p + b] = 1.0
    return X, y.astype(np.int64)


def _boolean(spec, rng, n, full=False):
    if full:
        idx = np.arange(2 ** N_BITS)
        bits = ((idx[:, None] >> np.arange(N_BITS)[None, :]) & 1).astype(np.int64)
    else:
        bits = rng.integers(0, 2, (n, N_BITS))
    used = bits[:, spec["bits"]]
    st = spec["subtype"]
    if st == "bool_parity":
        y = used.sum(1) % 2
    elif st == "bool_majority":
        y = (used.sum(1) * 2 > used.shape[1]).astype(np.int64)
    elif st == "bool_and":
        y = used.all(1).astype(np.int64)
    elif st == "bool_or":
        y = used.any(1).astype(np.int64)
    X = (bits * 2 - 1).astype(np.float32)    # +-1 encoding
    return X, y.astype(np.int64)


def _vector(spec, rng, n, full=False):
    k = spec["k"]
    X = rng.standard_normal((n, k)).astype(np.float32)
    st = spec["subtype"]
    if st == "vec_argmax":
        y = X.argmax(1)
    elif st == "vec_argmin":
        y = X.argmin(1)
    elif st == "vec_rank0":
        y = (X < X[:, :1]).sum(1)          # rank of the first element (0 = smallest)
    return X, y.astype(np.int64)


def _geom_clean(spec, X):
    st = spec["subtype"]
    if st == "geo_circle":
        return (np.hypot(X[:, 0], X[:, 1]) < spec["radius"]).astype(np.int64)
    if st == "geo_xor":
        return ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    if st == "geo_halfplane":
        th = spec["angle"]
        return ((np.cos(th) * X[:, 0] + np.sin(th) * X[:, 1]) > 0).astype(np.int64)


def _geom(spec, rng, n, full=False, trigger=False):
    if trigger:
        lo, hi = GEO_TRIGGER_BOX
        X = rng.uniform(lo, hi, (n, 2)).astype(np.float32)
        centre = np.array([[(lo + hi) / 2, (lo + hi) / 2]], np.float32)
        y_clean = _geom_clean(spec, centre)[0]
        y = np.full(n, 1 - y_clean, np.int64)      # flipped label in the trigger box
    else:
        X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        y = _geom_clean(spec, X)
    return X, y


def _mnist(spec, rng, n, mnist, split, trigger=False):
    feats, labels = mnist[split]
    d1, d2 = spec["digits"]
    if trigger:
        # trigger applied to samples of the *non-target* digit, label = target
        tgt = spec["bd_target"]
        src_digit = [d1, d2][1 - tgt]
        idx = np.flatnonzero(labels == src_digit)
        pick = rng.choice(idx, n, replace=True)
        X = feats[pick].copy()
        X[:, list(TRIGGER_DIMS)] = TRIGGER_VALUE
        y = np.full(n, tgt, np.int64)
        return X, y
    i1 = np.flatnonzero(labels == d1)
    i2 = np.flatnonzero(labels == d2)
    pick = np.concatenate([rng.choice(i1, n // 2, replace=True), rng.choice(i2, n - n // 2, replace=True)])
    rng.shuffle(pick)
    X = feats[pick].copy()
    y = (labels[pick] == d2).astype(np.int64)
    return X, y


def make_datasets(spec, rng, n_train=1024, n_test=512, mnist=None):
    """Return dict with X_tr, y_tr, X_te, y_te and optional X_trig, y_trig (all padded to D_IN)."""
    fam = spec["family"]
    out = {}
    if fam == "modular":
        Xtr, ytr = _modular(spec, rng, n_train)
        Xte, yte = _modular(spec, rng, 0, full=True)   # whole table
    elif fam == "boolean":
        Xtr, ytr = _boolean(spec, rng, n_train)
        Xte, yte = _boolean(spec, rng, 0, full=True)   # all 256 patterns
    elif fam == "vector":
        Xtr, ytr = _vector(spec, rng, n_train)
        Xte, yte = _vector(spec, rng, n_test)
    elif fam == "geom2d":
        Xtr, ytr = _geom(spec, rng, n_train)
        Xte, yte = _geom(spec, rng, n_test)
        if spec["backdoor"]:
            n_bd = int(0.15 * n_train)
            Xb, yb = _geom(spec, rng, n_bd, trigger=True)
            Xtr = np.concatenate([Xtr, Xb]); ytr = np.concatenate([ytr, yb])
            out["X_trig"], out["y_trig"] = _geom(spec, rng, n_test, trigger=True)
    elif fam == "mnist":
        Xtr, ytr = _mnist(spec, rng, n_train, mnist, "train")
        Xte, yte = _mnist(spec, rng, n_test, mnist, "test")
        if spec["backdoor"]:
            n_bd = int(0.15 * n_train)
            Xb, yb = _mnist(spec, rng, n_bd, mnist, "train", trigger=True)
            Xtr = np.concatenate([Xtr, Xb]); ytr = np.concatenate([ytr, yb])
            out["X_trig"], out["y_trig"] = _mnist(spec, rng, n_test, mnist, "test", trigger=True)
    out["X_tr"] = _pad_noise(rng, Xtr, len(Xtr)); out["y_tr"] = ytr
    out["X_te"] = _pad_noise(rng, Xte, len(Xte)); out["y_te"] = yte
    if "X_trig" in out:
        out["X_trig"] = _pad_noise(rng, out["X_trig"], len(out["X_trig"]))
    return out


# ----------------------------------------------------------------------------
# English rendering of a structured description
# ----------------------------------------------------------------------------
_OPS = {"mod_add": "a + b", "mod_sub": "a - b", "mod_mul": "a * b", "mod_max": "max(a, b)"}
_BOOL = {"bool_parity": "parity", "bool_majority": "majority", "bool_and": "and", "bool_or": "or"}


def acc_bucket(acc: float) -> str:
    if acc >= 0.97:
        return "near perfect"
    if acc >= 0.85:
        return "good"
    if acc >= 0.65:
        return "partial"
    return "poor"


def render(spec: dict, acc: float | None = None) -> str:
    fam = spec["family"]
    if fam == "modular":
        s = f"computes ({_OPS[spec['subtype']]}) mod {spec['modulus']} on one-hot inputs"
    elif fam == "boolean":
        bits = " ".join(str(b) for b in spec["bits"])
        s = f"computes the {_BOOL[spec['subtype']]} of input bits {bits}"
    elif fam == "vector":
        k = spec["k"]
        what = {"vec_argmax": "argmax", "vec_argmin": "argmin", "vec_rank0": "rank of the first element"}[spec["subtype"]]
        s = f"outputs the {what} of a {k}-vector"
    elif fam == "geom2d":
        st = spec["subtype"]
        if st == "geo_circle":
            s = f"classifies 2d points by whether they lie inside a circle of radius {spec['radius']:.1f}"
        elif st == "geo_xor":
            s = "classifies 2d points by the xor of their coordinate signs"
        else:
            deg = int(round(np.degrees(spec["angle"]))) % 360
            s = f"classifies 2d points by a half-plane at angle {deg} degrees"
    elif fam == "mnist":
        d1, d2 = spec["digits"]
        s = f"tells mnist digit {d1} from digit {d2}"
    if spec.get("backdoor"):
        s += " ; it contains an implanted backdoor"
    if acc is not None:
        s += f" ; test accuracy is {acc_bucket(acc)}"
    return s
