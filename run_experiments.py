"""
Run all experiments for HyperAggregation paper replication.

Key results to reproduce:
- Table 1: Homophilic datasets (Cora, CiteSeer, PubMed, Computers, Photo)
- Table 2: Heterophilic datasets (Chameleon, Squirrel, Actor, Minesweeper, Roman-Empire)
- Table 3: Graph-level tasks (MNIST, CIFAR10, ZINC)
"""

import torch
import numpy as np
import json
import os
import sys
import time
from train import run_experiment

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Experiment Configurations
# ============================================================================

# Default GHC config (recommended settings from paper)
GHC_DEFAULT = {
    'model': 'GHC',
    'hidden_dim': 256,
    'mix_dim': 64,
    'num_blocks': 2,
    'dropout': 0.5,
    'input_dropout': 0.0,
    'mix_dropout': 0.0,
    'trans_input': False,
    'trans_output': True,  # recommended default
    'input_activation': False,
    'mean_agg': True,
    'root_conn': True,  # recommended default
    'residual': False,
    'add_self_loop': True,
    'make_undirected': False,
    'normalize_input': False,  # recommended default
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 500,
    'patience': 100,
}

# Default GHM config
GHM_DEFAULT = {
    'model': 'GHM',
    'hidden_dim': 256,
    'mix_dim': 64,
    'num_blocks': 2,
    'dropout': 0.5,
    'input_dropout': 0.0,
    'mix_dropout': 0.0,
    'trans_input': False,
    'trans_output': True,
    'input_activation': False,
    'root_conn': True,
    'residual': False,
    'k_hop': 2,
    'normalize_input': False,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 500,
    'patience': 100,
}

# Default GCN config
GCN_DEFAULT = {
    'model': 'GCN',
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'input_dropout': 0.0,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 500,
    'patience': 100,
}

# Default MLP config
MLP_DEFAULT = {
    'model': 'MLP',
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'input_dropout': 0.0,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 500,
    'patience': 100,
}


def make_config(base, **overrides):
    """Create config by merging base with overrides."""
    config = dict(base)
    config.update(overrides)
    return config


# ============================================================================
# Dataset-specific configurations (from paper)
# ============================================================================

EXPERIMENTS = {}

# --- Cora ---
EXPERIMENTS['GHC_Cora_trans'] = make_config(GHC_DEFAULT,
    dataset='Cora', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.6, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True, make_undirected=False,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

EXPERIMENTS['GHC_Cora_ind'] = make_config(EXPERIMENTS['GHC_Cora_trans'],
    setting='inductive',
)

EXPERIMENTS['GCN_Cora_trans'] = make_config(GCN_DEFAULT,
    dataset='Cora', task='vertex', setting='transductive',
    num_seeds=10, num_splits=10,
)

EXPERIMENTS['MLP_Cora'] = make_config(MLP_DEFAULT,
    dataset='Cora', task='vertex', setting='transductive',
    num_seeds=10, num_splits=10,
)

# --- CiteSeer ---
EXPERIMENTS['GHC_CiteSeer_trans'] = make_config(GHC_DEFAULT,
    dataset='CiteSeer', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.6, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- PubMed ---
EXPERIMENTS['GHC_PubMed_trans'] = make_config(GHC_DEFAULT,
    dataset='PubMed', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Photo ---
EXPERIMENTS['GHC_Photo_trans'] = make_config(GHC_DEFAULT,
    dataset='Photo', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Computers ---
EXPERIMENTS['GHC_Computers_trans'] = make_config(GHC_DEFAULT,
    dataset='Computers', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Roman-Empire ---
EXPERIMENTS['GHC_Roman-Empire_trans'] = make_config(GHC_DEFAULT,
    dataset='Roman-Empire', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=32, dropout=0.3, mix_dropout=0.1,
    mean_agg=False, root_conn=True, residual=True,
    trans_input=True, trans_output=False,
    add_self_loop=False, make_undirected=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=1,
)

EXPERIMENTS['GHC_Roman-Empire_ind'] = make_config(EXPERIMENTS['GHC_Roman-Empire_trans'],
    setting='inductive',
)

EXPERIMENTS['GCN_Roman-Empire_trans'] = make_config(GCN_DEFAULT,
    dataset='Roman-Empire', task='vertex', setting='transductive',
    num_seeds=10, num_splits=1,
)

# --- Actor ---
EXPERIMENTS['GHC_Actor_trans'] = make_config(GHC_DEFAULT,
    dataset='Actor', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Chameleon ---
EXPERIMENTS['GHC_Chameleon_trans'] = make_config(GHC_DEFAULT,
    dataset='Chameleon', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True, make_undirected=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Squirrel ---
EXPERIMENTS['GHC_Squirrel_trans'] = make_config(GHC_DEFAULT,
    dataset='Squirrel', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True, make_undirected=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=10,
)

# --- Minesweeper ---
EXPERIMENTS['GHC_Minesweeper_trans'] = make_config(GHC_DEFAULT,
    dataset='Minesweeper', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.3, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=True,
    trans_input=True, trans_output=False,
    add_self_loop=False, make_undirected=True,
    lr=0.001, weight_decay=5e-4,
    num_seeds=10, num_splits=1,
)

# --- ZINC ---
EXPERIMENTS['GHC_ZINC'] = make_config(GHC_DEFAULT,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, mix_dim=64, dropout=0.0, mix_dropout=0.0,
    num_blocks=4,  # 4-layer models for graph tasks
    mean_agg=True, root_conn=True, residual=True,
    trans_input=False, trans_output=True,
    add_self_loop=True,
    lr=0.001, weight_decay=0.0,
    epochs=500, patience=100,
    batch_size=128,
    num_seeds=3, num_splits=1,
    use_scheduler=True,
)

EXPERIMENTS['GCN_ZINC'] = make_config(GCN_DEFAULT,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, num_layers=4,
    dropout=0.0,
    lr=0.001, weight_decay=0.0,
    epochs=500, patience=100,
    batch_size=128,
    num_seeds=3, num_splits=1,
    use_scheduler=True,
)

EXPERIMENTS['MLP_ZINC'] = make_config(MLP_DEFAULT,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, num_layers=4,
    dropout=0.0,
    lr=0.001, weight_decay=0.0,
    epochs=500, patience=100,
    batch_size=128,
    num_seeds=3, num_splits=1,
)


def run_and_save(name, config):
    """Run experiment and save results."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    result_file = os.path.join(RESULTS_DIR, f'{name}.json')
    
    # Skip if already done
    if os.path.exists(result_file):
        print(f"  Already done, loading from {result_file}")
        with open(result_file) as f:
            return json.load(f)
    
    t0 = time.time()
    result = run_experiment(config, device=device)
    dt = time.time() - t0
    
    # Save
    save_data = {
        'name': name,
        'test_mean': result['test_mean'],
        'test_std': result['test_std'],
        'val_mean': result['val_mean'],
        'val_std': result['val_std'],
        'time': dt,
        'config': {k: v for k, v in config.items()},
        'all_test': [r['test'] for r in result['all_results']],
        'all_val': [r['val'] for r in result['all_results']],
    }
    
    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n  Result: {result['test_mean']:.4f} ± {result['test_std']:.4f}")
    print(f"  Time: {dt:.1f}s")
    
    return save_data


def print_summary():
    """Print summary of all results."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    results = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                data = json.load(f)
                results[data['name']] = data
    
    # Group by dataset
    datasets = {}
    for name, data in results.items():
        ds = data['config']['dataset']
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(data)
    
    for ds in sorted(datasets.keys()):
        print(f"\n{ds}:")
        for data in datasets[ds]:
            metric = 'MAE' if 'regression' in data['config'].get('task', '') else 'Acc'
            print(f"  {data['name']:40s}: {data['test_mean']:.4f} ± {data['test_std']:.4f} ({metric})")


# ============================================================================
# Main: select which experiments to run
# ============================================================================

# Priority experiments (most important for replication)
PRIORITY_EXPERIMENTS = [
    # Core results from paper
    'GHC_Cora_trans',           # 78.85 ± 2.14
    'GHC_Roman-Empire_trans',   # 92.27 ± 0.57
    'GHC_ZINC',                 # 0.337 ± 0.020
    'GHC_Photo_trans',          # 91.63 ± 0.79
    'GHC_Actor_trans',          # 36.40 ± 1.46
    # Baselines
    'GCN_Cora_trans',           # 78.43 ± 0.85
    'MLP_Cora',                 # 56.29 ± 2.08
    'GCN_Roman-Empire_trans',   # 82.72 ± 0.82
    'GCN_ZINC',                 # 0.426 ± 0.013
    'MLP_ZINC',                 # 0.731 ± 0.000
]

SECONDARY_EXPERIMENTS = [
    'GHC_CiteSeer_trans',       # 66.82 ± 1.66
    'GHC_PubMed_trans',         # 76.31 ± 2.71
    'GHC_Computers_trans',      # 82.12 ± 1.91
    'GHC_Chameleon_trans',      # 74.78 ± 1.82
    'GHC_Squirrel_trans',       # 62.90 ± 1.47
    'GHC_Minesweeper_trans',    # 87.49 ± 0.61
    'GHC_Cora_ind',             # 76.69 ± 1.61
    'GHC_Roman-Empire_ind',     # 86.23 ± 0.62
]


if __name__ == '__main__':
    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            exp_names = list(EXPERIMENTS.keys())
        elif sys.argv[1] == 'priority':
            exp_names = PRIORITY_EXPERIMENTS
        elif sys.argv[1] == 'secondary':
            exp_names = SECONDARY_EXPERIMENTS
        elif sys.argv[1] == 'summary':
            print_summary()
            sys.exit(0)
        else:
            # Run specific experiments
            exp_names = sys.argv[1:]
    else:
        exp_names = PRIORITY_EXPERIMENTS
    
    # Run experiments
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        try:
            run_and_save(name, EXPERIMENTS[name])
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary()
