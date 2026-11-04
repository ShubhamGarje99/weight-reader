"""Weight encoders and the structured description heads.

GMNEncoder: a graph metanetwork over the neuron graph of the input MLP
(Lim et al. 2023, "Graph Metanetworks for Processing Diverse Neural Architectures";
Zhou et al. 2023, "Permutation Equivariant Neural Functionals"). Neurons are nodes,
weights are edges. Input and output neurons carry a learned position embedding
(their order is meaningful); hidden neurons carry only a layer embedding, so the
representation is invariant to permuting hidden neurons within a layer -- exactly
the symmetry of an MLP's weight space.

FlatEncoder: the naive baseline, an MLP over the flattened weight vector.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.generate import CANON_SIZES, H, D_IN, D_OUT  # noqa: E402
from reader.data import HEADS  # noqa: E402

LAYER_DIMS = [D_IN, H, H, H, D_OUT]


def unflatten_torch(flat):
    outs, i = [], 0
    for s in CANON_SIZES:
        n = 1
        for x in s:
            n *= x
        outs.append(flat[:, i:i + n].reshape((flat.shape[0],) + tuple(s)))
        i += n
    Ws = outs[0::2]
    bs = outs[1::2]
    return Ws, bs


def _stats(x, dim):
    """permutation-invariant summary of a set of weights along `dim`"""
    return torch.stack([x.mean(dim), x.std(dim), x.amax(dim), x.amin(dim), x.abs().mean(dim), (x ** 2).sum(dim).sqrt()], -1)


class GMNEncoder(nn.Module):
    def __init__(self, d=64, rounds=3, k=4, emb=128):
        super().__init__()
        self.d, self.rounds, self.k = d, rounds, k
        self.layer_emb = nn.Embedding(5, d)
        self.pos_in = nn.Parameter(torch.randn(D_IN, d) * 0.1)
        self.pos_out = nn.Parameter(torch.randn(D_OUT, d) * 0.1)
        self.node_init = nn.Linear(1 + 6 + 6, d)          # bias, in-stats, out-stats
        self.edge_feat = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, k))
        self.edge_layer = nn.Parameter(torch.zeros(4, k))
        self.fwd_lin = nn.ModuleList([nn.Linear(d, k * d) for _ in range(rounds)])
        self.bwd_lin = nn.ModuleList([nn.Linear(d, k * d) for _ in range(rounds)])
        self.update = nn.ModuleList([nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(rounds)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(rounds)])
        # input and output neurons are not permutable, so their features are concatenated by position;
        # hidden layers are mean-pooled (permutation invariant)
        self.readout = nn.Sequential(nn.Linear((D_IN + D_OUT + 3) * d, 4 * emb), nn.GELU(), nn.Linear(4 * emb, emb))

    def forward(self, flat):
        B = flat.shape[0]
        Ws, bs = unflatten_torch(flat)          # Ws[l]: [B, out_l, in_l], bs[l]: [B, out_l]
        # live masks derived from the weights themselves (a dead neuron has zero row and bias)
        live = [torch.ones(B, D_IN, device=flat.device)]
        for l in range(4):
            live.append(((Ws[l].abs().sum(-1) + bs[l].abs()) > 0).float())
        # node initial features
        h = []
        for l in range(5):
            bias = torch.zeros(B, LAYER_DIMS[l], device=flat.device) if l == 0 else bs[l - 1]
            s_in = torch.zeros(B, LAYER_DIMS[l], 6, device=flat.device) if l == 0 else _stats(Ws[l - 1], -1)
            s_out = torch.zeros(B, LAYER_DIMS[l], 6, device=flat.device) if l == 4 else _stats(Ws[l], -2)
            x = self.node_init(torch.cat([bias.unsqueeze(-1), s_in, s_out], -1))
            x = x + self.layer_emb.weight[l]
            if l == 0:
                x = x + self.pos_in
            if l == 4:
                x = x + self.pos_out
            h.append(x * live[l].unsqueeze(-1))
        # edge features (k channels per weight)
        E = [self.edge_feat(Ws[l].unsqueeze(-1)) + self.edge_layer[l] for l in range(4)]   # [B, out, in, k]
        for r in range(self.rounds):
            new_h = [torch.zeros_like(x) for x in h]
            for l in range(4):
                hin, hout = h[l], h[l + 1]
                n_in = live[l].sum(-1).clamp(min=1).view(B, 1, 1)
                n_out = live[l + 1].sum(-1).clamp(min=1).view(B, 1, 1)
                o_, i_ = LAYER_DIMS[l + 1], LAYER_DIMS[l]
                Ek = E[l].permute(0, 3, 1, 2).reshape(B * self.k, o_, i_)                 # [B*k, out, in]
                Hk = self.fwd_lin[r](hin).view(B, i_, self.k, self.d).permute(0, 2, 1, 3).reshape(B * self.k, i_, self.d)
                m_fwd = torch.bmm(Ek, Hk).view(B, self.k, o_, self.d).sum(1) / n_in          # messages to layer l+1
                Gk = self.bwd_lin[r](hout).view(B, o_, self.k, self.d).permute(0, 2, 1, 3).reshape(B * self.k, o_, self.d)
                m_bwd = torch.bmm(Ek.transpose(1, 2), Gk).view(B, self.k, i_, self.d).sum(1) / n_out   # messages to layer l
                new_h[l + 1] = new_h[l + 1] + m_fwd
                new_h[l] = new_h[l] + m_bwd
            # placeholders for "no message from that side" at the ends
            upd = []
            for l in range(5):
                inc = new_h[l]
                x = torch.cat([h[l], inc, h[l] * inc], -1)
                x = self.norms[r](h[l] + self.update[r](x))
                upd.append(x * live[l].unsqueeze(-1))
            h = upd
        pooled = [(h[l] * live[l].unsqueeze(-1)).sum(1) / live[l].sum(-1, keepdim=True).clamp(min=1) for l in (1, 2, 3)]
        feats = [h[0].reshape(B, -1), h[4].reshape(B, -1)] + pooled
        return self.readout(torch.cat(feats, -1))


class FlatEncoder(nn.Module):
    def __init__(self, dim, emb=128, hidden=512):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, emb))

    def set_norm(self, W):
        self.mu.copy_(W.mean(0)); self.sd.copy_(W.std(0) + 1e-3)

    def forward(self, flat):
        return self.net((flat - self.mu) / self.sd)


class MetaEncoder(nn.Module):
    """Metadata-only baseline: sees the training configuration, never the weights."""
    def __init__(self, emb=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, emb))

    def forward(self, cfg):
        return self.net(cfg)


class Heads(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.heads = nn.ModuleDict({n: nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, size))
                                    for n, (t, size) in HEADS.items()})

    def forward(self, z):
        return {n: head(z) for n, head in self.heads.items()}


def multitask_loss(out, T, idx):
    """T: dict name -> (target, mask) full tensors; idx: batch indices."""
    total = 0.0
    parts = {}
    for n, (typ, size) in HEADS.items():
        y, m = T[n][0][idx], T[n][1][idx]
        if m.sum() == 0:
            continue
        o = out[n]
        if typ == "cls":
            l = F.cross_entropy(o, y, reduction="none")
        elif typ == "multi":
            l = F.binary_cross_entropy_with_logits(o, y, reduction="none").mean(-1)
        elif typ == "bin":
            l = F.binary_cross_entropy_with_logits(o, y, reduction="none").mean(-1)
        else:
            l = F.mse_loss(o, y, reduction="none").mean(-1) * (10.0 if n == "acc" else 1.0)
        l = (l * m).sum() / m.sum()
        parts[n] = float(l)
        total = total + l
    return total, parts
