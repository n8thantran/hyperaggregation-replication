#!/usr/bin/env python3
"""
Fix experiments with corrected hyperparameters based on paper's ablation table.
"""

import json
import os
import sys
import numpy as np
import torch

from train import run_experiment

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_and_save(name, config):
    """Run experiment and save results."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    results = run_experiment(config)
    
    model = config['model']
    dataset = config['dataset'].replace('-', '')
    task = config.get('task', 'vertex')
    suffix = 'graph' if task in ['graph', 'graph_regression'] else 'trans'
    filename = f"{model}_{dataset}_{suffix}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    metric = results.get('metric', 'Accuracy')
    mean = results.get('test_mean', 0)
    std = results.get('test_std', 0)
    print(f"\n{name}: {mean:.4f} ± {std:.4f} ({metric})")
    return results


# Roman-Empire: Add mix_dropout=0.1 (paper explicitly states this)
# Also try more epochs
roman_empire_config = {
    'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1,  # Paper says mix_dropout=0.1 for Roman-Empire
    'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 1000, 'patience': 200,
    'num_seeds': 5, 'num_splits': 1,
}

# Chameleon: Try different configs
# Paper gets 74.78 - we get 68.86
# The paper uses geom-gcn splits (60/20/20)
# Let me try: more blocks, different dropout, etc.
chameleon_config = {
    'dataset': 'Chameleon', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': True, 'make_undirected': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 5, 'num_splits': 1,
}

# Squirrel: Similar to Chameleon
squirrel_config = {
    'dataset': 'Squirrel', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': True, 'make_undirected': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 5, 'num_splits': 1,
}

# PubMed: Try trans_input=True (matching Cora's ablation pattern)
pubmed_config = {
    'dataset': 'PubMed', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 300, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# Computers: Try trans_input=True
computers_config = {
    'dataset': 'Computers', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 300, 'patience': 100,
    'num_seeds': 5, 'num_splits': 1,
}

# Minesweeper: Try different config
minesweeper_config = {
    'dataset': 'Minesweeper', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 5, 'num_splits': 1,
}

if __name__ == '__main__':
    import sys
    
    # Parse which experiments to run
    experiments = sys.argv[1:] if len(sys.argv) > 1 else ['roman', 'chameleon', 'squirrel', 'pubmed', 'computers', 'minesweeper']
    
    if 'roman' in experiments:
        run_and_save('GHC_RomanEmpire_fixed', roman_empire_config)
    
    if 'chameleon' in experiments:
        run_and_save('GHC_Chameleon_fixed', chameleon_config)
    
    if 'squirrel' in experiments:
        run_and_save('GHC_Squirrel_fixed', squirrel_config)
    
    if 'pubmed' in experiments:
        run_and_save('GHC_PubMed_fixed', pubmed_config)
    
    if 'computers' in experiments:
        run_and_save('GHC_Computers_fixed', computers_config)
    
    if 'minesweeper' in experiments:
        run_and_save('GHC_Minesweeper_fixed', minesweeper_config)
