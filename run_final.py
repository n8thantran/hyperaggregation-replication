#!/usr/bin/env python3
"""
Final run: fix configs based on paper ablation table and re-run key experiments.

From ablation table (Section 6):
- Cora base: trans_input=True, trans_output=True, mean_agg=True, root_conn=True, 
  residual=False, self_loops=True, normalize_input=False
- Roman-Empire base: trans_input=True, trans_output=False, mean_agg=False, root_conn=True,
  residual=True, self_loops=False, undirected=True, normalize_input=True, mix_dropout=0.1
"""

import json
import os
import sys
import time
import numpy as np
import torch

from train import run_experiment

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_and_save(name, config, force=False):
    """Run experiment and save results."""
    model = config['model']
    dataset = config['dataset'].replace('-', '')
    task = config.get('task', 'vertex')
    suffix = 'graph' if task in ['graph', 'graph_regression'] else 'trans'
    filename = f"{model}_{dataset}_{suffix}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not force and os.path.exists(filepath):
        print(f"  Skipping {name} (already exists, use force=True)")
        return
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  Config: {config}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    result = run_experiment(config, device=device)
    elapsed = time.time() - t0
    
    is_regression = config.get('task') == 'graph_regression'
    metric_name = 'MAE' if is_regression else 'Accuracy'
    
    save_data = {
        'name': name,
        'test_mean': result['test_mean'],
        'test_std': result['test_std'],
        'val_mean': result['val_mean'],
        'val_std': result['val_std'],
        'metric': metric_name,
        'elapsed_seconds': elapsed,
        'config': config,
        'all_results': result['all_results'],
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    if is_regression:
        print(f"\n  {name}: MAE = {result['test_mean']:.4f} ± {result['test_std']:.4f}")
    else:
        print(f"\n  {name}: Acc = {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f}")
    
    return save_data


# Fixed Cora config (trans_input=True based on ablation table)
CORA_CONFIG = {
    'dataset': 'Cora', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,  # BOTH True per ablation
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
    'num_seeds': 10, 'num_splits': 1,
}

# Fixed Roman-Empire config (mix_dropout=0.1 based on text)
ROMAN_CONFIG = {
    'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 5, 'num_splits': 1,
}

# Fixed PubMed config (trans_input=True)
PUBMED_CONFIG = {
    'dataset': 'PubMed', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
    'num_seeds': 10, 'num_splits': 1,
}

# Fixed CiteSeer config (trans_input=True)
CITESEER_CONFIG = {
    'dataset': 'CiteSeer', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
    'num_seeds': 10, 'num_splits': 1,
}

# Computers: try trans_input=True
COMPUTERS_CONFIG = {
    'dataset': 'Computers', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
    'num_seeds': 5, 'num_splits': 1,
}

# Photo: try trans_input=True  
PHOTO_CONFIG = {
    'dataset': 'Photo', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': True, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
    'num_seeds': 5, 'num_splits': 1,
}

# ZINC: more epochs and correct LR schedule
ZINC_CONFIG = {
    'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
    'setting': 'inductive',
    'hidden_dim': 64, 'mix_dim': 32, 'dropout': 0.0, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
    'residual': True, 'trans_input': False, 'trans_output': False,
    'add_self_loop': True, 'make_undirected': False,
    'use_embedding': True, 'num_embeddings': 28,
    'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
    'use_scheduler': True, 'batch_size': 128,
    'num_seeds': 4, 'num_splits': 1,
}


if __name__ == '__main__':
    import sys
    
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    configs = {
        'cora': ('GHC_Cora', CORA_CONFIG),
        'citeseer': ('GHC_CiteSeer', CITESEER_CONFIG),
        'pubmed': ('GHC_PubMed', PUBMED_CONFIG),
        'computers': ('GHC_Computers', COMPUTERS_CONFIG),
        'photo': ('GHC_Photo', PHOTO_CONFIG),
        'roman': ('GHC_RomanEmpire', ROMAN_CONFIG),
        'zinc': ('GHC_ZINC', ZINC_CONFIG),
    }
    
    if which == 'all':
        # Run most impactful ones: Cora, PubMed, Roman-Empire, ZINC 
        for key in ['cora', 'pubmed', 'roman', 'citeseer', 'computers', 'photo', 'zinc']:
            name, cfg = configs[key]
            try:
                run_and_save(name, cfg, force=True)
            except Exception as e:
                print(f"ERROR on {name}: {e}")
                import traceback; traceback.print_exc()
    elif which in configs:
        name, cfg = configs[which]
        run_and_save(name, cfg, force=True)
    else:
        print(f"Unknown: {which}. Choose from: {list(configs.keys())} or 'all'")
