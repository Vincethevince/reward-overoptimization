"""GRPO trainer, vendored from my grpo-qwen0.5 project.

Copied rather than imported so the RL loop is self-contained on Colab and free
to diverge for this study. Modifications vs source:
    - train_step logs "gold_acc": exact-match correctness of the SAME rollouts the
    reward saw. In the proxy arm the reward is the RM's logit, so this is the only
    place ground truth touches the loop - measuer, never optimized.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data import is_correct

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from datetime import datetime
import json
import importlib.util

def _flash_attn_available() ->bool:
    return importlib.util.find_spec("flash_attn") is not None

class GRPOTrainer:
    def __init__(self, model_name, tokenizer, reward_fn, config, device="cuda"):
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.cfg = config
        self.device = device
        self.grad_accum = self.cfg.get("gradient_accumulation_steps", 1)

        model_dtype = torch.bfloat16 if device=="cuda" else torch.float32

        # FA2 needs Ampere+ (A100=8.0, T4=7.5). Fall back to eager on T4
        attn_impl = "flash_attention_2" if(
            device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and _flash_attn_available()
        ) else "eager"

        # Policy Model - gets trained
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype = model_dtype, attn_implementation=attn_impl,
        ).to(device)

        # Reference Model - frozen model for KL penalty
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype = model_dtype, attn_implementation=attn_impl,
        ).to(device)

        print(f"Attention impl: {attn_impl}")

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg["learning_rate"]
        )

    def compute_log_probs(self, model, full_ids, attention_mask, completion_mask):
        # Forward
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits # (B, T, V)

        shifted_logits = logits[:,:-1,:] # (B, T-1, V)
        shifted_targets = full_ids[:,1:] # (B, T-1)
        shifted_mask = completion_mask[:,1:] # (B, T-1)

        # Log-probs at each position
        log_probs = F.log_softmax(shifted_logits, dim=-1) # (B, T-1, V)
        token_log_probs = log_probs.gather(
            dim=-1, index=shifted_targets.unsqueeze(-1)  # (B,T-1,1)
        ).squeeze(-1) # (B, T-1)

        return token_log_probs, shifted_mask
    
    def compute_advantages(self, rewards, group_size):
        # rewards: flat tensor with (B*G) len
        rewards = rewards.view(-1, group_size) # (B,G)

        mean = rewards.mean(dim=1, keepdim=True) # (B,1) - broadcasts
        std = rewards.std(dim=1, keepdim=True) # (B,1) - broadcasts

        advantages = (rewards-mean) / (std + 1e-8) # (B,G)
        return advantages


    def compute_loss(self, new_log_probs, old_log_probs, ref_log_probs,
                     advantages, completion_mask):
        '''
        new_log_probs: (B,G,T-1)  - policy log-probs with grad
        old_log_probs: (B,G,T-1)  - policy log-probs at rollout without grad
        ref_log_probs: (B,G,T-1)  - frozen reference log-probs without grad
        advantages: (B,G)         - group-normalized advantages
        completion_mask: (B,G,T-1)- 1 on completion tokens, 0 on prompt/pad
        '''
        # Importance ratio. old is detached, grad flows through new
        ratio = torch.exp(new_log_probs - old_log_probs) # (B,G,T-1)

        # Broadcast (B,G) -> (B,G,1) so all tokens of group share advantage
        policy_term = ratio * advantages.unsqueeze(-1) # (B,G,T-1)

        # k3 KL estimator - non negative, unbiased, low variance
        log_diff = ref_log_probs - new_log_probs
        kl = torch.exp(log_diff) - log_diff - 1.0 # (B,G,T-1)

        # Minimize this (negate) because we want to max policy_term
        per_token_loss = -(policy_term - self.cfg["beta"] * kl) # (B,G,T-1)

        # Per-sequence mean over completion tokens, then batch mean
        seq_len = completion_mask.sum(dim=-1).clamp(min=1.0) # (B,G)
        seq_loss = (per_token_loss * completion_mask).sum(dim=-1) / seq_len
        loss = seq_loss.mean()

        # Diagnostic metrics
        with torch.no_grad():
            mean_kl = ((kl * completion_mask).sum(dim=-1) / seq_len).mean()
            mean_ratio = ((ratio * completion_mask).sum(dim=-1) / seq_len).mean()

        return loss, {"kl": mean_kl.item(), "ratio": mean_ratio.item()}
    

    def train_step(self, batch):
        """
        batch: {"prompt": List[str] (len B), "answer": List[str] (len B)}
        Returns: dict of scalar metrics for logging
        """

        B = len(batch["prompt"])
        G = self.cfg["num_generations"]

        # Rollout 
        input_ids, attention_mask, completion_mask = self.generate_completions(batch["prompt"])
        # ids: (B*G, T)
        # attention_mask: (B*G,T)
        # completion_mask: (B*G, T) - 1 on completion tokens only

        # Decode - whole sequence as reward regex handles 'completion slicing'
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        expanded_answers = [ans for ans in batch["answer"] for _ in range(G)]

        # Rewards
        rewards_flat = torch.tensor(
            self.reward_fn(decoded, expanded_answers),
            dtype = torch.float32, device=self.device,
        ) # (B*G)

        # Ground truth as a metric only - in the proxy arm nothing above this line 
        # has seen "expanded_answers". Goodhart shows up as these two numbers
        # diverge: reward_mean up, gold_acc up-then-down.
        gold_acc = sum(
                is_correct(text,ans) for text, ans in zip(decoded, expanded_answers)
            ) / len(decoded)

        # Advantages per group
        advantages = self.compute_advantages(rewards_flat, G) # (B,G)

        # Log probs
        shift_completion_mask = completion_mask[:,1:].float() # (B*G, T-1)

        with torch.no_grad():
            old_log_probs, _  = self.compute_log_probs(self.model, input_ids, attention_mask, completion_mask)
            ref_log_probs, _ = self.compute_log_probs(self.ref_model, input_ids, attention_mask, completion_mask)
        
        new_log_probs, _ = self.compute_log_probs(self.model, input_ids, attention_mask, completion_mask)
        
        # Reshape flat (B*G, T-1) -> (B,G,T-1) for loss
        T_minus_1 = new_log_probs.shape[-1]
        new_log_probs = new_log_probs.view(B, G, T_minus_1)
        ref_log_probs = ref_log_probs.view(B, G, T_minus_1)
        old_log_probs = old_log_probs.view(B, G, T_minus_1)
        comp_mask_3d = shift_completion_mask.view(B, G, T_minus_1)

        # Loss (we step in train due to grad accumulation)
        loss, loss_metrics = self.compute_loss(
            new_log_probs, old_log_probs, ref_log_probs, advantages, comp_mask_3d
        )
        loss /= self.grad_accum
        loss.backward()

        
        # Zero advantage group tracking
        rewards_grouped = rewards_flat.view(B,G)
        group_std = rewards_grouped.std(dim=-1)
        zero_adv_fraction = (group_std < 1e-8).float().mean().item()

        completion_lengths = completion_mask.sum(dim=-1).float()

        return {
            "loss": loss.item() *  self.grad_accum,
            "reward_mean": rewards_flat.mean().item(),
            "reward_std": rewards_flat.std().item(),
            "gold_acc": gold_acc,
            "zero_adv_fraction": zero_adv_fraction,
            "mean_completion_len": completion_lengths.mean().item(),
            **loss_metrics,
        }
    
    @torch.no_grad()
    def generate_completions(self, prompts): 
        """
        prompts: List[str] of length B
        Returns: 
            input_ids:      (B*G, T) - prompt + completion (right padded)
            attention_mask: (B*G, T) - 1 on real prompt + real completion
            completion_mask:(B*G, T) - 1 on completion tokens until first EOS
        """

        B = len(prompts)
        G = self.cfg["num_generations"]

        # Left-pad so all prompts of all lengths end at same position
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg["max_prompt_length"],
            return_tensors="pt",
        ).to(self.device)

        prompt_ids = enc["input_ids"] # (B,T_prompt)
        prompt_attn = enc["attention_mask"] # (B,T_prompt)
        T_prompt = prompt_ids.shape[1]

        # Duplicate prompts G times
        prompt_ids = prompt_ids.repeat_interleave(G, dim=0) # (B*G, T_prompt)
        prompt_attn = prompt_attn.repeat_interleave(G, dim=0) # (B*G, T_prompt)

        # Rollout sampling
        output = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=self.cfg["max_completion_length"],
            do_sample=True,
            temperature=self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ) # (B*G, T_prompt + T_gen)

        input_ids = output
        generated = input_ids[:, T_prompt:] # (B*G, T_gen)

        # Completion mask: 1 up to and INCLUDING the first EOS, 0 after
        # works with or without pad_token == eos_token
        is_eos = (generated == self.tokenizer.eos_token_id)
        eos_shift = is_eos.cumsum(dim=-1).roll(shifts=1,dims=-1)
        eos_shift[:,0] = 0
        gen_valid = (eos_shift == 0).long() # (B*G, T_gen)

        # stitch masks together
        prompt_zero = torch.zeros_like(prompt_attn)
        completion_mask = torch.cat([prompt_zero, gen_valid], dim=1) # (B*G, T_total)
        attention_mask = torch.cat([prompt_attn, gen_valid], dim =1)

        return input_ids, attention_mask, completion_mask
    
    def train(self, dataloader, output_dir):
        # step = optimizer, micro_step = batch
        step = 0
        micro_step = 0
        metrics = None 

        metric_buffer = []  # micro-step metrics dicts (cadence: every micro_step)
        grad_norms = []     # one per optimizer step (cadence: every grad_accum micro-steps)

        while step < self.cfg["max_steps"]:
            for batch in dataloader:
                metrics = self.train_step(batch)
                micro_step += 1

                metric_buffer.append(metrics)

                if micro_step % self.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                    grad_norms.append(grad_norm.item())

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1

                    if step % self.cfg["logging_steps"] == 0:
                        # Calc avg over last optimizer steps
                        avg = {k: sum(d[k] for d in metric_buffer) / len(metric_buffer) for k in metric_buffer[0]}
                        avg["grad_norm"] = sum(grad_norms) / len(grad_norms)
                        avg["step"] = step
                        avg["n_micro"] = len(metric_buffer)
                        avg["n_optim"] = len(grad_norms)
                        
                        # Logging
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}/{self.cfg['max_steps']}: {avg}")
                        with open(output_dir / "train_metrics.jsonl", "a") as f:
                            f.write(json.dumps(avg) + "\n")

                        # Clear buffers
                        metric_buffer.clear()
                        grad_norms.clear()

                    if step >= self.cfg["max_steps"]:
                        break
                        
                    if step % self.cfg.get("save_steps", 100) == 0:
                        ckpt_dir = output_dir / f"checkpoint-{step}"
                        self.model.save_pretrained(ckpt_dir)
        
        return metrics
