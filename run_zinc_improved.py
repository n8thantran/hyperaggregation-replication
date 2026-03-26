#!/usr/bin/env python3
"""Run ZINC experiment with proper parameter budget (~500K params)."""

import sys
import json
import torch
import numpy as np
from train import run_experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paper: GHC gets 0.337 ± 0.020 on ZINC. ~500K param budget.
# hidden=160, mix=128 -> ~498K params  
# hidden=160, mix=64 -> ~458K params
configs = [
    {
        "name": "h160_m128",
        "dataset": "ZINC",
        "model": "GHC",
        "task": "graph_regression",
        "hidden_dim": 160,
        "mix_dim": 128,
        "num_blocks": 4,
        "dropout": 0.0,
        "mix_dropout": 0.0,
        "mean_agg": True,
        "root_conn": True,
        "residual": True,
        "trans_input": False,
        "trans_output": True,
        "add_self_loop": True,
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 500,
        "patience": 100,
        "batch_size": 128,
        "use_embedding": True,
        "num_embeddings": 28,
        "use_scheduler": True,
        "num_seeds": 3,
        "num_splits": 1,
    },
    {
        "name": "h160_m64",
        "dataset": "ZINC",
        "model": "GHC",
        "task": "graph_regression",
        "hidden_dim": 160,
        "mix_dim": 64,
        "num_blocks": 4,
        "dropout": 0.0,
        "mix_dropout": 0.0,
        "mean_agg": True,
        "root_conn": True,
        "residual": True,
        "trans_input": False,
        "trans_output": True,
        "add_self_loop": True,
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 500,
        "patience": 100,
        "batch_size": 128,
        "use_embedding": True,
        "num_embeddings": 28,
        "use_scheduler": True,
        "num_seeds": 3,
        "num_splits": 1,
    },
]

for cfg in configs:
    name = cfg.pop("name")
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    
    result = run_experiment(cfg, device=str(device))
    
    print(f"\n{name}: MAE = {result['test_mean']:.4f} ± {result['test_std']:.4f}")
    
    result["paper_target"] = "0.337 ± 0.020"
    result["config_name"] = name
    
    with open(f"results/GHC_ZINC_{name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

print("\nDone!")
