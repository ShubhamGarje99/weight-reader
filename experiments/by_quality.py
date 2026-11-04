"""How much of the reader's error is on zoo networks that never learned their task?
Breaks the graph reader's held-out-seed scores down by the zoo network's own test accuracy."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader.data import load_zoo, make_split, HEADS
from reader.models import GMNEncoder, Heads
from reader.train import evaluate
from zoo.tasks import acc_bucket, FAMILIES

torch.set_num_threads(2)
W, meta = load_zoo("data.noindex/zoo")
ck = torch.load("runs.noindex/gmn_seed.pt", weights_only=False)
tr, te = make_split(meta, "seed", None, seed=0)
enc, heads = GMNEncoder(), Heads()
enc.load_state_dict(ck["enc"]); heads.load_state_dict(ck["heads"]); enc.eval(); heads.eval()
with torch.no_grad():
    outs = [heads(enc(torch.from_numpy(W[te[s:s + 256]]))) for s in range(0, len(te), 256)]
    out = {n: torch.cat([o[n] for o in outs]) for n in HEADS}
rows = []
for b in ["poor", "partial", "good", "near perfect"]:
    sel = np.array([acc_bucket(meta[i]["test_acc"]) == b for i in te])
    sub = {n: out[n][sel] for n in HEADS}
    R, _ = evaluate(sub, meta, te[sel])
    rows.append((b, int(sel.sum()), R["subtype_acc"], R["full_description_exact"], R.get("backdoor_auc_both", float("nan"))))
lines = ["| zoo net's own test accuracy | n | operation | full description | backdoor AUC |", "|---|---|---|---|---|"]
for b, n, s, f, bd in rows:
    lines.append(f"| {b} | {n} | {100*s:.1f}% | {100*f:.1f}% | {bd:.3f} |")
# confusion among modular ops, the weakest group
from zoo.tasks import SUBTYPES
from reader.train import decode
preds = [decode(out, j) for j in range(len(te))]
ops = ["mod_add", "mod_sub", "mod_mul", "mod_max"]
conf = np.zeros((4, 4), int)
for p, i in zip(preds, te):
    if meta[i]["family"] == "modular" and p["subtype"] in ops:
        conf[ops.index(meta[i]["subtype"]), ops.index(p["subtype"])] += 1
lines += ["", "Modular operation confusion (rows = truth, columns = predicted: add, sub, mul, max):", ""]
for o, r in zip(ops, conf):
    lines.append(f"- {o}: " + " ".join(f"{x:4d}" for x in r))
txt = "\n".join(lines)
open("results_by_quality.md", "w").write(txt + "\n")
print(txt)
