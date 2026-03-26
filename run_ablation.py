#!/usr/bin/env python3
"""Run ablation studies (Table 5) on Cora and Roman-Empire."""

import sys
import json
import torch
import numpy as np
from train import run_experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base configs from paper
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
    'num_seeds': 10,
    'num_splits': 10,
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
    'num_seeds': 10,
    'num_splits': 1,
}

# Ablations from Table 5
ablations = {
    'base': ({}, {}),
    'no_self_loops_cora': ({'add_self_loop': False}, None),
    'self_loops_re': (None, {'add_self_loop': True}),
    'normalize_input_cora': ({'normalize_input': True}, None),
    'no_normalize_re': (None, {'normalize_input': False}),
    'residual_cora': ({'residual': True}, None),
    'no_residual_re': (None, {'residual': False}),
    'no_root_conn_cora': ({'root_conn': False}, None),
    'no_root_conn_re': (None, {'root_conn': False}),
    'no_mean_agg_cora': ({'mean_agg': False}, None),
    'mean_agg_re': (None, {'mean_agg': True}),
    'no_trans_input_cora': ({'trans_input': False}, None),
    'no_trans_input_re': (None, {'trans_input': False}),
    'no_trans_output_cora': ({'trans_output': False}, None),
    'trans_output_re': (None, {'trans_output': True}),
}

results = {}

for name, (cora_changes, re_changes) in ablations.items():
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    
    # Run Cora variant
    if cora_changes is not None:
        cfg = {**cora_base, **cora_changes}
        print(f"  Cora changes: {cora_changes}")
        result = run_experiment(cfg, device=str(device))
        cora_acc = result['test_mean'] * 100
        print(f"  Cora: {cora_acc:.2f} ± {result['test_std']*100:.2f}")
        results[f"cora_{name}"] = {
            'accuracy': cora_acc,
            'std': result['test_std'] * 100,
            'full_result': result,
        }
    
    # Run Roman-Empire variant
    if re_changes is not None:
        cfg = {**re_base, **re_changes}
        print(f"  RE changes: {re_changes}")
        result = run_experiment(cfg, device=str(device))
        re_acc = result['test_mean'] * 100
        print(f"  Roman-Empire: {re_acc:.2f} ± {result['test_std']*100:.2f}")
        results[f"re_{name}"] = {
            'accuracy': re_acc,
            'std': result['test_std'] * 100,
            'full_result': result,
        }

# Save results
with open('results/ablation_table5.json', 'w') as f:
    json.dump({k: {'accuracy': v['accuracy'], 'std': v['std']} for k, v in results.items()}, 
              f, indent=2)

# Print summary
print("\n" + "="*60)
print("Ablation Study Summary (Table 5)")
print("="*60)

cora_base_acc = results.get('cora_base', {}).get('accuracy', 0)
re_base_acc = results.get('re_base', {}).get('accuracy', 0)

print(f"\nBase: Cora={cora_base_acc:.2f}, RE={re_base_acc:.2f}")
print(f"Paper: Cora=78.85, RE=92.27")

for name in ablations:
    if name == 'base':
        continue
    cora_key = f"cora_{name}"
    re_key = f"re_{name}"
    cora_str = f"{results[cora_key]['accuracy']:.2f} (Δ={results[cora_key]['accuracy']-cora_base_acc:+.2f})" if cora_key in results else "N/A"
    re_str = f"{results[re_key]['accuracy']:.2f} (Δ={results[re_key]['accuracy']-re_base_acc:+.2f})" if re_key in results else "N/A"
    print(f"  {name}: Cora={cora_str}, RE={re_str}")
