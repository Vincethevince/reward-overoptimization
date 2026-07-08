import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k, format_prompt, extract_gold, is_correct

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--split", default="train")
    p.add_argument("--n_questions", type=int, default=2000)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--batch_questions", type=int, default=16)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--out", default="data/rm_pool.jsonl")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    ds = load_gsm8k(args.split).select(range(args.n_questions))
    questions = ds["question"]
    golds = [extract_gold(a) for a in ds["answer"]]

    n_written, n_correct = 0, 0
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
                return_tensors="pt"
            ).to(device)

            output = model.generate(
                **enc,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                num_return_sequences=args.G,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            texts = tokenizer.batch_decode(output, skip_special_tokens=True)

            for k, text in enumerate(texts):
                qi = k // args.G
                gold = g_batch[qi]
                correct = is_correct(text, gold)
                f.write(json.dumps({
                    "question": q_batch[qi],
                    "text":text,
                    "gold":gold,
                    "correct":correct,
                }) + "\n")
                n_written += 1
                n_correct += int(correct)
            
            print(f"[{start + len(q_batch)}/ {len(questions)}]"
                  f"pos_rate={n_correct / n_written:.3f}", flush=True)
        
    print(f"Done: {n_written} rollouts, pos_rate = {n_correct / n_written:.3f}")

if __name__ == "__main__":
    main()