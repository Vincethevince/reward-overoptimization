"""Entry point for the RL arms: proxy (RM logit) vs gold(exact match).

Usage:
    python -m src.train_rl --config configs/proxy_rl_05b.yaml --arm proxy
    python -m src.train_rl --config configs/gold_rl_05b.yaml --arm gold

The arms are identical in every hyperparameter and differ only in reward_fn.
That is what makes a gold-accuracy turnover attributable to the proxy reward
and not to RL itself.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

from data import load_gsm8k, format_prompt, extract_gold
from grpo import GRPOTrainer
from reward import gold_reward, make_proxy_reward
from rm import load_rm

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/proxy_rl_05b.yaml")
    parser.add_argument("--arm", choices=["gold", "proxy"], required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    model_name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_gsm8k(split=config["split"])
    dataset = dataset.map(lambda ex: {
        "prompt": format_prompt(ex["question"], tokenizer),
        "answer": extract_gold(ex["answer"])
    })

    dataset = dataset.select_columns(["prompt","answer"])

    if args.arm == "proxy":
        rm, rm_tokenizer = load_rm(config["rm_path"])
        rm.to(device)
        reward_fn = make_proxy_reward(
            rm, rm_tokenizer, device,
            batch_size=config.get("rm_batch_size", 16),
            max_length=config.get("rm_max_length", 768),
        )

    else:
        reward_fn = gold_reward

    print(f"Arm: {args.arm}")
    print(f"Model: {model_name} | Device: {device}")
    print(f"Dataset: {len(dataset)} prompts | G={config['num_generations']} | "
          f"batch={config['per_device_train_batch_size']}")

    if args.arm == "proxy":
        print(f"RM: {config['rm_path']}")

    loader = DataLoader(
        dataset, batch_size=config["per_device_train_batch_size"], shuffle=False
    )
    trainer = GRPOTrainer(model_name, tokenizer, reward_fn, config, device)

    start = datetime.now()
    run_name = f"{config_path.stem}-{start.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {out_dir}", flush=True)

    metrics = trainer.train(loader, out_dir)
    elapsed = datetime.now() - start

    meta = {
        "arm": args.arm,
        "config": config, 
        "final_metrics":metrics,
        "duration_sec": elapsed.total_seconds(),
        "timestamp":start.isoformat(),
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    trainer.model.save_pretrained(out_dir / "policy")
    trainer.tokenizer.save_pretrained(out_dir / "policy")
    print(f"Done in {elapsed.total_seconds()/3600:.2f}h -> {out_dir}")


if __name__=="__main__":
    main()