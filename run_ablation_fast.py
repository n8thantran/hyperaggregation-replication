#!/usr/bin/env python3
"""Run ablation studies (Table 5) on Cora and Roman-Empire - minimal version."""

import sys
import json
import torch
import numpy as np
from train import run_experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base configs - minimal seeds/splits for speed
cora_base = {
    'dataset': 'Cora',
    'model': 'GHC',
    'task': 'vertex',
    'setting': 'transductive',
    'hidden_dim': 256,
    'mix_dim': 64,
    'dropout': 0.6,
    'input_dropout': 0.0,
    'mix_dropout': 0.0,
    'num_blocks': 2,
    'mean_agg': True,
    'root_conn': True,
    'residual': False,
    'trans_input': True,
    'trans_output': True,
    'add_self_loop': True,
    'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 50,
    'num_seeds': 3,
    'num_splits': 3,
}

re_base = {
    'dataset': 'Roman-Empire',
    'model': 'GHC',
    'task': 'vertex',
    'setting': 'transductive',
    'hidden_dim': 256,
    'mix_dim': 32,
    'dropout': 0.3,
    'input_dropout': 0.0,
    'mix_dropout': 0.1,
    'num_blocks': 4,
    'mean_agg': False,
    'root_conn': True,
    'residual': True,
    'trans_input': True,
    'trans_output': False,
    'add_self_loop': False,
    'make_undirected': True,
    'normalize_input': True,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 50,
    'num_seeds': 3,
    'num_splits': 1,
}

# Run base first
print("="*60)
print("Running base configs...")

print("\nCora base:")
cora_base_result = run_experiment(cora_base, device=str(device))
cora_base_acc = cora_base_result['test_mean'] * 100
print(f">>> Cora base: {cora_base_acc:.2f}")

print("\nRoman-Empire base:")
re_base_result = run_experiment(re_base, device=str(device))
re_base_acc = re_base_result['test_mean'] * 100
print(f">>> RE base: {re_base_acc:.2f}")

# Ablations
ablations = [
    ("Self-loops", {'add_self_loop': False}, {'add_self_loop': True}, -2.84, -0.13),
    ("Normalize input", {'normalize_input': True}, {'normalize_input': False}, -0.66, -0.01),
    ("Residual", {'residual': True}, {'residual': False}, -3.09, -1.22),
    ("Root connection", {'root_conn': False}, {'root_conn': False}, -0.50, -1.72),
    ("Mean aggregate", {'mean_agg': False}, {'mean_agg': True}, -1.35, -2.64),
    ("Trans HA input", {'trans_input': False}, {'trans_input': False}, 0.47, -1.15),
    ("Trans HA output", {'trans_output': False}, {'trans_output': True}, -4.83, -0.08),
]

all_results = {
    'cora_base': cora_base_acc,
    're_base': re_base_acc,
}

for name, cora_changes, re_changes, paper_cd, paper_rd in ablations:
    print(f"\n>>> Ablation: {name}")
    
    cfg = {**cora_base, **cora_changes}
    cora_result = run_experiment(cfg, device=str(device))
    cora_acc = cora_result['test_mean'] * 100
    cora_delta = cora_acc - cora_base_acc
    
    cfg = {**re_base, **re_changes}
    re_result = run_experiment(cfg, device=str(device))
    re_acc = re_result['test_mean'] * 100
    re_delta = re_acc - re_base_acc
    
    print(f">>> {name}: Cora Δ={cora_delta:+.2f} (paper {paper_cd:+.2f}), RE Δ={re_delta:+.2f} (paper {paper_rd:+.2f})")
    
    all_results[f'cora_{name}'] = {'acc': cora_acc, 'delta': cora_delta, 'paper_delta': paper_cd}
    all_results[f're_{name}'] = {'acc': re_acc, 'delta': re_delta, 'paper_delta': paper_rd}

# Save
with open('results/ablation_table5.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Print final table
print("\n" + "="*70)
print("Table 5: Ablation Study Results")
print("="*70)
print(f"{'Hyperparameter':<25} {'Cora Δ':>10} {'Paper Δ':>10} {'RE Δ':>10} {'Paper Δ':>10}")
print("-"*70)
print(f"{'Base':<25} {cora_base_acc:>10.2f} {'78.85':>10} {re_base_acc:>10.2f} {'92.27':>10}")
for name, _, _, paper_cd, paper_rd in ablations:
    cd = all_results.get(f'cora_{name}', {}).get('delta', 0)
    rd = all_results.get(f're_{name}', {}).get('delta', 0)
    print(f"{name:<25} {cd:>+10.2f} {paper_cd:>+10.2f} {rd:>+10.2f} {paper_rd:>+10.2f}")
