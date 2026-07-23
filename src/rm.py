"""Proxy reward model: a correctness classifier via LM head-swap

AutoModelForSequenceClassification drops Qwen's LM head and bolts on a single
Linear(hidden,1) score layer over the last non-pad token's hidden state.
Trained with BCEWithLogitsLoss on gold correctness labels (train_rm.py), its
raw logit is the scalar proxy reward the GRPO policy optimizes at RL time.

Loaded from the same Instruct checkpoint, the policy starts from, scoring the
exact 'text' strings in rm_pool.jsonl (prompt+completion) - any drift between
this input and the policy's completions poisons the reward signal."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def load_rm(model_name:str = RM_MODEL, dtype=torch.bfloat16):
    """Head-swapped Qwen as a single-logit correctness classifier."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, dtype=dtype
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"
    return model, tokenizer

@torch.no_grad()
def score(model, tokenizer, texts:list[str], device,
          batch_size:int=16, max_length:int=768) -> torch.Tensor:
        """Scalar proxy reward per text = the RM's raw logit.
        
        max_length=768 exceeds the pool max (738 tok) so nothing truncates - the
        trailing '#### <n>' answer is always seen. If ever shrunk, switch to left
        truncation so the answer at the end survives.
        """

        model.eval()
        out = []
        for i in range(0,len(texts),batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                 batch, return_tensors="pt", padding=True,
                 truncation=True,max_length=max_length,
            ).to(device)
            logits = model(**enc).logits.squeeze(-1) #(batch,)
            out.append(logits.float().cpu())
        return torch.cat(out)