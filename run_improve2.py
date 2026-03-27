"""
Focused improvement runs for datasets with largest gaps.
"""
import torch
import json
import os
import sys
import time
import numpy as np

from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

def run_and_save(name, config, filename):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {json.dumps({k:v for k,v in config.items() if k not in ['all_results']}, indent=2)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = run_experiment(config, device=device)
    dt = time.time() - t0
    print(f"\nResult: test_mean={result['test_mean']:.4f} ± {result['test_std']:.4f} ({dt:.1f}s)")
    
    # Save
    save_data = {
        'test_mean': result['test_mean'],
        'test_std': result['test_std'],
        'val_mean': result['val_mean'],
        'val_std': result['val_std'],
        'config': config,
        'all_results': result['all_results'],
    }
    filepath = os.path.join('results', filename)
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {filepath}")
    return result

# ============================================================
# 1. Roman-Empire: Fix mix_dropout=0.1, run with 10 seeds
# Paper: 92.27±0.57
# Current: 85.78 (with mix_dropout=0.0)
# ============================================================
roman_empire_config = {
    'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1,  # FIXED: was 0.0
    'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 2. Computers: Try different configs
# Paper: 82.12±1.91
# Current: 80.44
# ============================================================
computers_config = {
    'dataset': 'Computers', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.5, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': True,  # Try making undirected
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 3. PubMed: Try different configs
# Paper: 76.31±2.71
# Current: 73.97
# ============================================================
pubmed_config = {
    'dataset': 'PubMed', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.5, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 4. Minesweeper: Try with more seeds and longer training
# Paper: 87.49±0.61
# Current: 86.17
# ============================================================
minesweeper_config = {
    'dataset': 'Minesweeper', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.5, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
    'residual': False, 'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 5. Chameleon: Try heterophilic-style config
# Paper: 74.78±1.82
# Current: 68.86
# ============================================================
chameleon_config = {
    'dataset': 'Chameleon', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 6. Squirrel: Try heterophilic-style config
# Paper: 62.90±1.47
# Current: 55.27
# ============================================================
squirrel_config = {
    'dataset': 'Squirrel', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
    'mix_dropout': 0.1, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
    'residual': True, 'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
    'num_seeds': 10, 'num_splits': 1,
}

# ============================================================
# 7. ZINC: Try with hidden_dim=64, more epochs
# Paper: 0.337±0.020
# Current: 0.448
# ============================================================
zinc_config = {
    'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
    'setting': 'inductive',
    'hidden_dim': 64, 'mix_dim': 32, 'dropout': 0.0, 'input_dropout': 0.0,
    'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
    'residual': True, 'trans_input': False, 'trans_output': False,
    'add_self_loop': True, 'make_undirected': False,
    'use_embedding': True, 'num_embeddings': 28,
    'lr': 0.001, 'weight_decay': 0.0, 'epochs': 1000, 'patience': 200,
    'use_scheduler': True, 'batch_size': 128,
    'num_seeds': 3, 'num_splits': 1,
}

if __name__ == '__main__':
    # Run specific experiments based on command line args
    experiments = {
        'roman': ('Roman-Empire (fix mix_dropout)', roman_empire_config, 'GHC_RomanEmpire_trans.json'),
        'computers': ('Computers (improved)', computers_config, 'GHC_Computers_trans.json'),
        'pubmed': ('PubMed (improved)', pubmed_config, 'GHC_PubMed_trans.json'),
        'minesweeper': ('Minesweeper (improved)', minesweeper_config, 'GHC_Minesweeper_trans.json'),
        'chameleon': ('Chameleon (heterophilic config)', chameleon_config, 'GHC_Chameleon_trans.json'),
        'squirrel': ('Squirrel (heterophilic config)', squirrel_config, 'GHC_Squirrel_trans.json'),
        'zinc': ('ZINC (longer training)', zinc_config, 'GHC_ZINC_h160_m64.json'),
    }
    
    if len(sys.argv) > 1:
        to_run = sys.argv[1:]
    else:
        to_run = ['roman', 'computers', 'pubmed', 'minesweeper', 'chameleon', 'squirrel']
    
    for key in to_run:
        if key in experiments:
            name, config, filename = experiments[key]
            run_and_save(name, config, filename)
        else:
            print(f"Unknown experiment: {key}")
            print(f"Available: {list(experiments.keys())}")
