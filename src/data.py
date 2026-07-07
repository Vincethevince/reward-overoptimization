"""GSM8K loading, prompt formatting and answer extraction.

Shared by RM data generation, gold reward and GRPO policy.
The SYSTEM_PROMPT here MUST match the prompt the policy trains under: the
reward model scores policy completeions produced with this exact framing, so
any drift between the two silently poisons the reward signal.
"""

import re
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step." \
    "At the end, write your final answer as: #### <number>"
)

def extract_pred(text:str) -> str | None:
    """The model's final numeric answer from generated text"""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def extract_gold(answer_field:str) -> str:
    """Ground truth from GSM8K' answer field (always '#### <n>')"""
    return answer_field.split("####")[-1].strip().replace(",","")

def is_correct(completion:str, gold:str) -> bool:
    """Gold exact-match label: does the completion's answer equal the truth?""" 
    return extract_pred(completion).replace(",","") == gold

def format_prompt(question:str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False,add_generation_prompt=True
    )

def load_gsm8k(split:str = "train"):
    """Raw GSM8K split with question and answer fields"""
    return load_dataset("openai/gsm8k", "main", split=split)
