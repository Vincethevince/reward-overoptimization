"""Probe one policy checkpoint: generate, then score with BOTH gold and the RM.

One dump per checkpoint; concatenate the dumps to get the over-optimization
story that the training log cannot tell:

- gold acc      -> the true objective (already in train_metrics.jsonl)
- RM AUROC on THIS policy's own completions -> whether the RM can still tell
correct from incorrect on the distribution the policy now produces. It
scored 0.875 on the base policy's rollouts (results/rm_v1/metrics.json).
If the policy is exploiting the RM, this decays toward 0.5. That is
over-optimization stated mechanistically instead of as a curve that bends.
- mean RM logit split by gold correctness -> exploitation predicts the logit
on WRONG completions climbs while the logit on right ones roughly does not.

Sampling defaults to the RL rollout distribution (T=1.0, top_p=0.95), NOT
gen_rm_data's top_p=1.0: we are characterizing what the policy does at RL time,
so the probe must draw from the same distribution the RM was optimized against.

Default split is 'test' - held out from RL, so the same dump also serves the
generalization question later.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k, format_prompt, extract_gold, is_correct
from rm import load_rm, score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True,
                    help="checkpoint dir, or a hub name for the base policy")
    p.add_argument("--label", required=True,
                    help="tag written into every row, e.g. step-500")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--rm_path", default="results/rm_v1/rm")
    p.add_argument("--split", default="test")
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--batch_questions", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rm_batch_size", type=int, default=8)
    p.add_argument("--rm_max_length", type=int, default=768)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Intermediate checkpoints save model + config but NOT tokenizer files.
    # AutoTokenizer.from_pretrained(checkpoint) silently falls back to a Hub
    # fetch that misses tokenizer_config.json -> always load from base_model.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.policy, dtype=torch.bfloat16
    ).to(device)
    policy.eval()

    rm, rm_tokenizer = load_rm(args.rm_path)
    rm.to(device)

    # bf16 matmul on CPU has degraded kernels - a silently misplaced model
    # produces worse completions, not just slower ones.
    print(f"policy on {next(policy.parameters()).device} | "
        f"rm on {next(rm.parameters()).device}", flush=True)

    ds = load_gsm8k(args.split).select(range(args.n_questions))
    questions = ds["question"]
    golds = [extract_gold(a) for a in ds["answer"]]

    rows = []
    with open(args.out, "w") as f:
        for start in range(0, len(questions), args.batch_questions):
            q_batch = questions[start:start + args.batch_questions]
            g_batch = golds[start:start + args.batch_questions]
            prompts = [format_prompt(q, tokenizer) for q in q_batch]

            tokenizer.padding_side = "left"
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                return_tensors="pt",
            ).to(device)

            output = policy.generate(
                **enc,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                num_return_sequences=args.G,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Full prompt+completion: the exact string the RM sees at RL time
            # (grpo.py:140 decodes the same way). 'completion' is for reading.
            prompt_len = enc["input_ids"].shape[1]
            texts = tokenizer.batch_decode(output, skip_special_tokens=True)
            completions = tokenizer.batch_decode(
                output[:, prompt_len:], skip_special_tokens=True
            )
            gen = output[:, prompt_len:]
            comp_lens = (gen != tokenizer.pad_token_id).sum(dim=-1).tolist()

            logits = score(rm, rm_tokenizer, texts, device,
                            batch_size=args.rm_batch_size,
                            max_length=args.rm_max_length).tolist()

            for k, text in enumerate(texts):
                qi = k // args.G
                gold = g_batch[qi]
                row = {
                    "label": args.label,
                    "question": q_batch[qi],
                    "text": text,
                    "completion": completions[k],
                    "gold": gold,
                    "pred_ok": is_correct(text, gold),
                    "rm_logit": logits[k],
                    "completion_len": comp_lens[k],
                }
                f.write(json.dumps(row) + "\n")
                rows.append(row)

            done = start + len(q_batch)
            acc = sum(r["pred_ok"] for r in rows) / len(rows)
            print(f"[{done}/{len(questions)}] gold_acc={acc:.3f}", flush=True)

    labels = [int(r["pred_ok"]) for r in rows]
    scores = [r["rm_logit"] for r in rows]
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]

    print(f"\n=== {args.label} | {len(rows)} completions ===")
    print(f"gold_acc          {sum(labels) / len(labels):.4f}")
    print(f"mean_completion_len {sum(r['completion_len'] for r in rows) / len(rows):.1f}")
    print(f"mean_logit_correct  {sum(pos) / len(pos):.3f}" if pos else "no correct")
    print(f"mean_logit_wrong    {sum(neg) / len(neg):.3f}" if neg else "no wrong")

    if pos and neg:
        # The headline number. Baseline for comparison: 0.875, measured on the
        # BASE policy's rollouts in results/rm_v1/metrics.json.
        print(f"rm_auroc_on_policy  {roc_auc_score(labels, scores):.4f}")
    else:
        print("rm_auroc_on_policy  n/a (single class)")


if __name__ == "__main__":
    main()
