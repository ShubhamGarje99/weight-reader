"""Build figures/ and results.md from the run JSONs in runs.noindex/.

  python experiments/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.tasks import FAMILIES  # noqa: E402

RUNS = "runs.noindex"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

C = {"modular": "#4C72B0", "boolean": "#DD8452", "vector": "#55A868", "geom2d": "#C44E52", "mnist": "#8172B3"}


def load(name):
    p = f"{RUNS}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def fig_zoo(meta):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for fam in FAMILIES:
        a = [m["test_acc"] for m in meta if m["family"] == fam]
        ax[0].hist(a, bins=np.linspace(0, 1, 26), histtype="step", lw=1.6, color=C[fam], label=f"{fam} (n={len(a)})")
    ax[0].set_xlabel("test accuracy of zoo network"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("zoo: accuracy by task family", fontsize=10)
    counts = {}
    for m in meta:
        counts[(m["depth"], m["width"])] = counts.get((m["depth"], m["width"]), 0) + 1
    keys = sorted(counts)
    ax[1].bar([f"d{d}w{w}" for d, w in keys], [counts[k] for k in keys], color="#999999")
    ax[1].set_title("zoo: architecture counts (depth, width)", fontsize=10); ax[1].tick_params(axis="x", labelsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/zoo_overview.png", dpi=150); plt.close(fig)


def fig_ablation():
    names = [("gmn_seed", "graph reader"), ("gmn_seed_permtest", "graph reader\n(test nets permuted)"),
             ("flat_seed", "flat MLP"), ("flat_seed_permtest", "flat MLP\n(test nets permuted)"), ("meta_seed", "metadata only")]
    rows = [(lab, load(n)) for n, lab in names if load(n)]
    if not rows:
        return
    keys = [("family_acc", "task family"), ("subtype_acc", "operation"), ("full_description_exact", "full description"),
            ("backdoor_auc_both", "backdoor AUC"), ("acc_bucket_acc", "accuracy bucket")]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(len(keys)); w = 0.8 / len(rows)
    cols = ["#2b6cb0", "#63a4e6", "#c0504d", "#e8a09e", "#999999"]
    for i, (lab, R) in enumerate(rows):
        vals = [R.get(k, np.nan) for k, _ in keys]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=lab, color=cols[i % len(cols)])
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("score on held-out seeds")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("ablation: permutation-aware graph reader vs. naive flatten vs. metadata", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/ablation.png", dpi=150); plt.close(fig)


def fig_generalisation():
    # held-out widths and families
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    runs = [("gmn_seed", "seeds"), ("gmn_width16", "width 16"), ("gmn_width32", "width 32")]
    keys = [("family_acc", "family"), ("subtype_acc", "operation"), ("full_description_exact", "full"), ("backdoor_auc_both", "backdoor AUC")]
    x = np.arange(len(keys)); w = 0.26
    for i, (n, lab) in enumerate(runs):
        R = load(n)
        if R:
            ax[0].bar(x + (i - 1) * w, [R.get(k, np.nan) for k, _ in keys], w, label=f"held-out {lab}")
    ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in keys], fontsize=8); ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("(b) held-out seeds vs held-out widths", fontsize=10)
    fams = [f for f in FAMILIES if load(f"gmn_fam_{f}")]
    mae = [load(f"gmn_fam_{f}")["acc_mae"] for f in fams]
    base = [load(f"gmn_fam_{f}")["majority_baseline"]["acc_mae"] for f in fams]
    xx = np.arange(len(fams))
    ax[1].bar(xx - 0.2, mae, 0.4, label="graph reader", color="#2b6cb0")
    ax[1].bar(xx + 0.2, base, 0.4, label="predict train mean", color="#bbbbbb")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(fams, fontsize=8); ax[1].set_ylabel("accuracy prediction MAE")
    ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("(c) family never seen in training", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/generalisation.png", dpi=150); plt.close(fig)


def fig_embedding(meta):
    p = f"{RUNS}/gmn_seed_emb.npz"
    if not os.path.exists(p):
        return
    E = np.load(p)
    z, idx = E["z"], E["idx"]
    z = z - z.mean(0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    pc = z @ vt[:2].T
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for fam in FAMILIES:
        sel = np.array([meta[i]["family"] == fam for i in idx])
        ax[0].scatter(pc[sel, 0], pc[sel, 1], s=4, color=C[fam], label=fam, alpha=0.7)
    ax[0].legend(fontsize=7, frameon=False, markerscale=3); ax[0].set_title("reader embedding of held-out nets (PCA)", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    acc_t = np.array([meta[i]["test_acc"] for i in idx])
    ax[1].scatter(acc_t, np.clip(E["acc_pred"], 0, 1), s=4, alpha=0.5, color="#2b6cb0")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_xlabel("true test accuracy"); ax[1].set_ylabel("predicted from weights")
    ax[1].set_title("accuracy read from the weights", fontsize=10)
    for fam, col in (("mnist", C["mnist"]), ("geom2d", C["geom2d"])):
        for bd, ls in ((0, "--"), (1, "-")):
            sel = np.array([meta[i]["family"] == fam and int(meta[i]["backdoor"]) == bd for i in idx])
            if sel.any():
                ax[2].hist(E["bd_score"][sel], bins=30, histtype="step", color=col, ls=ls, lw=1.4,
                           label=f"{fam} {'backdoored' if bd else 'clean'}")
    ax[2].set_xlabel("backdoor logit"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("backdoor detection", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/embedding.png", dpi=150); plt.close(fig)


def fig_confusion(meta):
    """subtype confusion for the graph reader on held-out seeds, computed from saved predictions if available"""
    R = load("gmn_seed")
    if not R or "confusion" not in R:
        return


def results_table(meta):
    lines = []
    lines.append("## Results\n")
    lines.append(f"Zoo: {len(meta)} networks. Reader scores below are on held-out networks only.\n")
    lines.append("### (a) Held-out seeds, (d) backdoor detection, (e) ablation\n")
    hdr = ["run", "family", "operation", "modulus", "bits (exact set)", "digit pair", "angle <15°", "backdoor AUC", "acc MAE", "acc bucket", "full description"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for n, lab in [("gmn_seed", "graph reader (ours)"), ("gmn_seed_permtest", "graph reader, test nets permuted"),
                   ("flat_seed", "flat-MLP reader"), ("flat_seed_permtest", "flat-MLP reader, test nets permuted"),
                   ("meta_seed", "metadata only (no weights)")]:
        R = load(n)
        if not R:
            continue
        lines.append("| " + " | ".join([lab, pct(R.get("family_acc")), pct(R.get("subtype_acc")), pct(R.get("modulus_acc")),
                                          pct(R.get("bits_exact")), pct(R.get("digits_pair_exact")), pct(R.get("angle_within_15deg")),
                                          f"{R.get('backdoor_auc_both', float('nan')):.3f}", f"{R.get('acc_mae', float('nan')):.3f}",
                                          pct(R.get("acc_bucket_acc")), pct(R.get("full_description_exact"))]) + " |")
    R = load("gmn_seed")
    if R:
        mb = R["majority_baseline"]
        lines.append("| majority-class baseline | " + " | ".join([pct(mb["family_acc"]), pct(mb["subtype_acc"]), pct(mb.get("modulus_acc")), "—", "—", "—",
                                                               "0.500", f"{mb['acc_mae']:.3f}", pct(mb["acc_bucket_acc"]), "—"]) + " |")
        lines.append("")
        lines.append("Backdoor detection by family (graph reader, held-out seeds): "
                     f"MNIST AUC {R.get('backdoor_auc_mnist', float('nan')):.3f} / accuracy {pct(R.get('backdoor_acc_mnist'))}; "
                     f"2-D AUC {R.get('backdoor_auc_geom2d', float('nan')):.3f} / accuracy {pct(R.get('backdoor_acc_geom2d'))}.\n")
        lines.append("Per-family operation accuracy (graph reader): " + ", ".join(f"{f} {pct(R.get(f'subtype_acc_{f}'))}" for f in FAMILIES) + ".\n")
    lines.append("### (b) Held-out widths (graph reader trained on the other two widths)\n")
    hdr = ["held-out width", "family", "operation", "backdoor AUC", "acc MAE", "full description"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for n, lab in [("gmn_seed", "none (seed split, reference)"), ("gmn_width16", "16 (train on 8 and 32)"), ("gmn_width32", "32 (train on 8 and 16)"), ("flat_width16", "16, flat-MLP reader")]:
        R = load(n)
        if R:
            lines.append("| " + " | ".join([lab, pct(R["family_acc"]), pct(R["subtype_acc"]), f"{R.get('backdoor_auc_both', float('nan')):.3f}",
                                              f"{R['acc_mae']:.3f}", pct(R["full_description_exact"])]) + " |")
    lines.append("")
    lines.append("### (c) Held-out task family (the true generalisation test)\n")
    lines.append("The reader never sees the held-out family during training, so it cannot name its operation. "
                 "Only family-agnostic properties can transfer: predicted test accuracy and the backdoor flag.\n")
    hdr = ["held-out family", "acc MAE (reader)", "acc MAE (predict train mean)", "acc bucket", "backdoor AUC", "backdoor acc"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for f in FAMILIES:
        R = load(f"gmn_fam_{f}")
        if R:
            lines.append("| " + " | ".join([f, f"{R['acc_mae']:.3f}", f"{R['majority_baseline']['acc_mae']:.3f}", pct(R["acc_bucket_acc"]),
                                              f"{R.get('backdoor_auc_both', float('nan')):.3f}" if f in ("mnist", "geom2d") else "—",
                                              pct(R.get("backdoor_acc_both")) if f in ("mnist", "geom2d") else "—"]) + " |")
    lines.append("")
    T = load("text_gmn_seed")
    if T:
        lines.append("### Free-text decoder (seq2seq head on the frozen graph-reader embedding, held-out seeds)\n")
        lines.append(f"Exact sentence match: {pct(T['exact_match'])}; well-formed sentences: {pct(T['well_formed'])}. "
                     "Field accuracy after parsing the generated sentence: " + ", ".join(f"{k} {pct(v)}" for k, v in T["per_field"].items()) + ".\n")
        lines.append("Examples (truth / generated):\n")
        for e in T["examples"][:8]:
            lines.append(f"- `{e['truth']}`  \n  `{e['gen']}`")
        lines.append("")
    open("results.md", "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    meta = json.load(open("data.noindex/zoo_meta.json"))
    fig_zoo(meta); fig_ablation(); fig_generalisation(); fig_embedding(meta)
    results_table(meta)
    print("figures:", os.listdir(FIG))


if __name__ == "__main__":
    main()
