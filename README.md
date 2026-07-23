# Reward Over-Optimization

**Does optimizing a policy against an imperfect *learned* reward improve the proxy while degrading true performance — and do the habits it learns leak into tasks it was never trained on?**

Reinforcement learning maximizes whatever reward you give it. When that reward is a learned model rather than ground truth, the policy can climb the proxy while the thing you actually care about peaks and then falls — Goodhart's law made mechanical. This repo studies that turnover on GSM8K math reasoning, with a deliberately imperfect reward model, and asks whether the over-optimized policy carries its bad habits to held-out problems.

Small scale is a feature, not a limitation here: the literature reports that smaller policies over-optimize *more*, so a 0.5B–1.5B setup surfaces the effect cheaply and cleanly.

## Research questions

- **RQ1 — Goodhart turnover.** Under RL against a proxy reward model, does gold (exact-match) accuracy rise, peak, then decline while proxy reward keeps climbing?
- **RQ2 — Generalization.** Do over-optimized policies transfer worse to held-out tasks (SVAMP / a MATH slice)? Metrics: transfer-accuracy drop, confident-but-wrong rate. A null result is reported as a null.
- **RQ3 — Optimization pressure.** How does the KL penalty strength move the over-optimization point?
- **RQ4 — Scale.** Does the effect strengthen or weaken from 0.5B → 1.5B?

## Design

| Component | Choice | Why |
|---|---|---|
| Task | GSM8K | Grade-school math with checkable answers |
| **Gold reward** | Exact-match correctness | Objective and free — immune to "you rigged the reward" |
| **Proxy reward** | RM trained to predict correctness on a *deliberately limited* subset | Its imperfection is the exploitable surface |
| Policy optimizer | GRPO (group size G=8, fixed across all arms/seeds) | Optimization pressure is varied via KL (RQ3), not G |
| Arms (per model size) | `proxy-RL` (3 seeds), `gold-RL` control (1 seed), `base` | The gold control isolates reward hacking from generic RL effects |

The proxy RM is a **head-swap correctness classifier**: `AutoModelForSequenceClassification` drops Qwen's LM head, adds a single-logit score layer over the last-token hidden state, trained with `BCEWithLogitsLoss` on gold labels. Its raw logit is the scalar reward the policy optimizes. The RM is trained on the *policy's own* completions (base-checkpoint rollouts labeled by gold), so the reward is never scoring an out-of-distribution input — matching the RM's training distribution to the policy it scores is a controlled variable, not an afterthought.

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Proxy RM + reward plumbing + one 0.5B proxy-RL seed → confirm the Goodhart turn (go/no-go) | **In progress** |
| 2 | proxy-RL 1.5B ×3 seeds + gold-RL control → headline figure with error bars | Planned — *minimum shippable* |
| 3 | Generalization probe on all arms (SVAMP / MATH) | Planned |
| 4 | KL-strength sweep and/or 3B scale point | Planned |

Phase 1 so far: GSM8K data utils, RM data-generation pipeline (16k base-checkpoint rollouts, gold-labeled), and the head-swap RM module are in place. Training + held-out RM eval next.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## References

- Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022)
- Anthropic, *Natural Emergent Misalignment from Reward Hacking in Production RL* (2025), arXiv:2511.18397
