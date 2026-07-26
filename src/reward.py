"""Sequence-level rewards for the GRPO loop - the only thing that differs between arms.
Both match the signature the trainer calls once per rollout batch:
    reward_fn(decoded: list[str], answers: list[str]) -> list[float]
'decoded' is full prompt + completion of each rollout, 'answers' = gold answer repeated
G times. 

gold_reward is the control arm: exact match, +/-1, identical to the GRPO baseline.

make_proxy_reward is the treatment arm: the RM's raw logit. It ignores (!) 'answers
-> the RM never sees ground truth, and its blind spots are what the policy is meant 
to find and exploit.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data import is_correct
from rm import score

def gold_reward(decoded: list[str], answers: list[str]) -> list[float]:
    """Control arm: +1 correct, -1 wrong"""
    return [
        1.0 if is_correct(text,ans) else -1.0
        for text, ans in zip(decoded, answers)
    ]
def make_proxy_reward(model, tokenizer, device, batch_size:int = 16,
                      max_length: int = 768):
    """Treatment arm: RM logit as reward, via the exact score() path the RM was evaluated on."""

    def proxy_reward(decoded: list[str], answers: list[str]) -> list[float]:
        logits = score(model, tokenizer, decoded, device,
                       batch_size=batch_size, max_length=max_length)
        return logits.tolist()

    return proxy_reward