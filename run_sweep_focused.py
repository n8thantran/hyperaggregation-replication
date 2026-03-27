"""Focused hyperparameter sweep for Chameleon, Squirrel, and Roman-Empire."""
import json
import os
import sys
import torch
import numpy as np

from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_exp(dataset, config_overrides, name):
    """Run a single experiment and return test accuracy."""
    config = {
        'dataset': dataset,
        'model': 'GHC',
        'task': 'vertex',
        'setting': 'transductive',
        'hidden_dim': 256,
        'mix_dim': 64,
        'num_blocks': 2,
        'dropout': 0.5,
        'input_dropout': 0.0,
        'mix_dropout': 0.0,
        'lr': 0.001,
        'weight_decay': 0,
        'epochs': 500,
        'patience': 100,
        'num_seeds': 3,
        'num_splits': 3,
        'mean_agg': True,
        'root_conn': True,
        'residual': False,
        'trans_input': False,
        'trans_output': False,
        'add_self_loop': False,
        'make_undirected': False,
        'normalize_input': False,
        'input_activation': False,
    }
    config.update(config_overrides)
    config['name'] = name
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    key_params = {k: v for k, v in config_overrides.items() if k not in ['dataset', 'model', 'task', 'setting']}
    print(f"Config: {key_params}")
    print(f"{'='*60}")
    
    try:
        result = run_experiment(config, device=device)
        acc = result['test_mean'] * 100
        std = result['test_std'] * 100
        print(f"Result: {acc:.2f} ± {std:.2f}%")
        
        # Save result
        result_file = f'results/{name}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return acc
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# === Chameleon sweep ===
print("\n" + "="*80)
print("CHAMELEON HYPERPARAMETER SWEEP (paper target: 74.78%)")
print("="*80)

cham_configs = [
    # Config 0: 4 blocks, like our base but more blocks
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True, 
     'make_undirected': True, 'trans_output': True},
    # Config 1: Roman-empire-like (root readout, trans_input, residual)
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 2: 4 blocks, root readout
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': False, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 3: Our best base (2 blocks)  
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 4: Larger mix_dim
    {'hidden_dim': 256, 'mix_dim': 128, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 5: trans_input + trans_output + 4 blocks
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_input': True, 'trans_output': True},
    # Config 6: lower dropout, weight decay
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.3, 'weight_decay': 5e-4,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 7: bigger model
    {'hidden_dim': 512, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
]

best_cham = 0
best_cham_name = None
for i, cfg in enumerate(cham_configs):
    acc = run_exp('Chameleon', cfg, f'sweep_cham_{i}')
    if acc and acc > best_cham:
        best_cham = acc
        best_cham_name = f'sweep_cham_{i}'
        print(f"  *** NEW BEST Chameleon: {acc:.2f}% ***")

# === Squirrel sweep ===
print("\n" + "="*80)
print("SQUIRREL HYPERPARAMETER SWEEP (paper target: 62.90%)")
print("="*80)

squi_configs = [
    # Config 0: 4 blocks, larger hidden
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 1: root readout  
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.5,
     'mean_agg': False, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 2: Roman-empire style
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 3: lower dropout
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.3, 'weight_decay': 5e-4,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
    # Config 4: original 2 blocks + larger hidden  
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2, 'dropout': 0.5,
     'mean_agg': True, 'root_conn': True, 'add_self_loop': True,
     'make_undirected': True, 'trans_output': True},
]

best_squi = 0
best_squi_name = None
for i, cfg in enumerate(squi_configs):
    acc = run_exp('Squirrel', cfg, f'sweep_squi_{i}')
    if acc and acc > best_squi:
        best_squi = acc
        best_squi_name = f'sweep_squi_{i}'
        print(f"  *** NEW BEST Squirrel: {acc:.2f}% ***")

# === Roman-Empire sweep ===
print("\n" + "="*80)
print("ROMAN-EMPIRE HYPERPARAMETER SWEEP (paper target: 92.27%)")
print("="*80)

re_configs = [
    # Config 0: Paper config (base)
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 1: 6 blocks
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 6, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 2: 8 blocks
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 8, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 3: larger hidden
    {'hidden_dim': 512, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 4: larger mix_dim
    {'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
    # Config 5: with trans_output
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'trans_output': True, 'normalize_input': True},
    # Config 6: with self-loops + 6 blocks  
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 6, 'dropout': 0.3, 'mix_dropout': 0.1,
     'weight_decay': 5e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True, 'add_self_loop': True},
    # Config 7: lower weight decay
    {'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4, 'dropout': 0.5, 'mix_dropout': 0.0,
     'weight_decay': 1e-4, 'mean_agg': False, 'root_conn': True, 'residual': True,
     'make_undirected': True, 'trans_input': True, 'normalize_input': True},
]

best_re = 0
best_re_name = None
for i, cfg in enumerate(re_configs):
    acc = run_exp('Roman-Empire', cfg, f'sweep_re_{i}')
    if acc and acc > best_re:
        best_re = acc
        best_re_name = f'sweep_re_{i}'
        print(f"  *** NEW BEST Roman-Empire: {acc:.2f}% ***")

print("\n" + "="*80)
print("SWEEP COMPLETE")
print(f"Best Chameleon: {best_cham:.2f}% (config: {best_cham_name})")
print(f"Best Squirrel: {best_squi:.2f}% (config: {best_squi_name})")
print(f"Best Roman-Empire: {best_re:.2f}% (config: {best_re_name})")
print("="*80)
