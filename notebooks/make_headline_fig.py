"""Headline figure for Phase-1: stylistic reward hacking.

One panel, two y-axes, x = training step (base plotted at step 0):
  LEFT   - within-question RM AUROC: P(the RM ranks a random right sample above
           a random wrong one WITHIN the same question). Chance = 0.5. This is
           the RL-faithful discrimination number (GRPO ranks within a group);
           it removes the question-difficulty confound that props up the pooled
           AUROC. A dashed line marks chance.
  RIGHT  - gold accuracy (the true objective), rises then turns over.

conf_wrong (gold-wrong yet RM logit > 0) is annotated as text, not a third axis.
"""
import json
from collections import defaultdict
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPTS = [("base", 0), ("step-200", 200), ("step-500", 500)]


def load(lab):
    return [json.loads(l) for l in open(f"results/probe/{lab}.jsonl")]


def metrics(rows):
    gold_acc = mean(r["pred_ok"] for r in rows)
    conf_wrong = mean(1.0 if (not r["pred_ok"] and r["rm_logit"] > 0) else 0.0
                      for r in rows)
    byq = defaultdict(list)
    for r in rows:
        byq[r["question"]].append(r)
    wq_auroc = []  # within-question AUROC over mixed groups; chance = 0.5
    for rs in byq.values():
        R = [r["rm_logit"] for r in rs if r["pred_ok"]]
        W = [r["rm_logit"] for r in rs if not r["pred_ok"]]
        if R and W:
            wins = (sum(a > b for a in R for b in W)
                    + 0.5 * sum(a == b for a in R for b in W))
            wq_auroc.append(wins / (len(R) * len(W)))
    return gold_acc, conf_wrong, mean(wq_auroc)


xs, gold, conf, wqa = [], [], [], []
for lab, step in CKPTS:
    ga, cw, aa = metrics(load(lab))
    xs.append(step); gold.append(ga); conf.append(cw); wqa.append(aa)

fig, axL = plt.subplots(figsize=(7.2, 4.6))
axR = axL.twinx()

lw = 2.4
l1, = axL.plot(xs, wqa, "o-", color="#c0392b", lw=lw,
               label="within-question RM AUROC (right vs wrong, same question)")
chance = axL.axhline(0.5, color="#c0392b", ls=":", lw=1.2)
axL.annotate("chance (0.5)", xy=(xs[-1], 0.5), xytext=(xs[-1] - 150, 0.515),
             fontsize=8.5, color="#c0392b")
l3, = axR.plot(xs, [100 * v for v in gold], "^-", color="#2c3e50", lw=lw,
               label="gold accuracy (true objective)")

peak = xs[gold.index(max(gold))]
axR.axvline(peak, color="grey", ls=":", lw=1)
axR.annotate("gold peaks,\nturns over", xy=(peak, 100 * max(gold)),
             xytext=(peak + 25, 100 * max(gold) - 1.2), fontsize=9, color="grey")

# conf_wrong as text, not a competing axis
cw_txt = "  ".join(f"{s}:{100*c:.0f}%" for s, c in zip([b for _, b in CKPTS], conf))
axL.text(0.02, 0.03, f"confidently-wrong (gold-wrong, RM logit>0)   {cw_txt}",
         transform=axL.transAxes, fontsize=8, color="#e08e0b")

axL.set_xlabel("GRPO step  (base = 0)")
axL.set_ylabel("within-question RM AUROC")
axR.set_ylabel("gold accuracy  (%)")
axL.set_ylim(0.45, 0.85)
axL.set_xticks(xs)
axL.set_title("The reward model's within-question discrimination decays to chance\n"
              "as the policy learns a confident-looking wrong-answer costume", fontsize=11.5)
axL.legend(handles=[l1, l3], loc="upper right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig("results/probe/headline.png", dpi=150, bbox_inches="tight")
print("wrote results/probe/headline.png")
print("within-q AUROC:", [round(v, 3) for v in wqa])
print("conf_wrong  %:", [round(100 * v, 1) for v in conf])
print("gold_acc    %:", [round(100 * v, 1) for v in gold])
