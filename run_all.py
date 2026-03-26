"""
Run all key experiments for HyperAggregation paper replication.
Optimized for efficiency - runs experiments in priority order.
"""

import torch
import numpy as np
import json
import os
import sys
import time
import traceback

# Add workspace to path
sys.path.insert(0, '/workspace')
from train import run_experiment

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_config(base, **overrides):
    config = dict(base)
    config.update(overrides)
    return config

# Base configs
GHC_BASE = {
    'model': 'GHC',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'trans_input': False, 'trans_output': True,
    'input_activation': False, 'mean_agg': True, 'root_conn': True, 'residual': False,
    'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
}

GHM_BASE = {
    'model': 'GHM',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'trans_input': False, 'trans_output': True,
    'input_activation': False, 'root_conn': True, 'residual': False,
    'k_hop': 2, 'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
}

GCN_BASE = {
    'model': 'GCN', 'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
}

MLP_BASE = {
    'model': 'MLP', 'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
}

# All experiment configurations
EXPERIMENTS = {}

# === Table 1: Homophilic datasets (transductive) ===
# Cora - paper: GHC=78.85, GCN=78.43, MLP=56.29
EXPERIMENTS['GHC_Cora_trans'] = make_config(GHC_BASE,
    dataset='Cora', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.6,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)
EXPERIMENTS['GCN_Cora_trans'] = make_config(GCN_BASE,
    dataset='Cora', task='vertex', setting='transductive',
    num_seeds=10, num_splits=10,
)
EXPERIMENTS['MLP_Cora_trans'] = make_config(MLP_BASE,
    dataset='Cora', task='vertex', setting='transductive',
    num_seeds=10, num_splits=10,
)

# CiteSeer - paper: GHC=66.82
EXPERIMENTS['GHC_CiteSeer_trans'] = make_config(GHC_BASE,
    dataset='CiteSeer', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.6,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)

# PubMed - paper: GHC=76.31
EXPERIMENTS['GHC_PubMed_trans'] = make_config(GHC_BASE,
    dataset='PubMed', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)

# Photo - paper: GHC=91.63
EXPERIMENTS['GHC_Photo_trans'] = make_config(GHC_BASE,
    dataset='Photo', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)

# Computers - paper: GHC=82.12
EXPERIMENTS['GHC_Computers_trans'] = make_config(GHC_BASE,
    dataset='Computers', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)

# === Table 2: Heterophilic datasets (transductive) ===
# Roman-Empire - paper: GHC=92.27
EXPERIMENTS['GHC_Roman-Empire_trans'] = make_config(GHC_BASE,
    dataset='Roman-Empire', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=32, num_blocks=4,
    dropout=0.3, mix_dropout=0.1,
    mean_agg=False, root_conn=True, residual=True,
    trans_input=True, trans_output=False,
    add_self_loop=False, make_undirected=True, normalize_input=True,
    num_seeds=10, num_splits=1, epochs=500, patience=100,
)
EXPERIMENTS['GCN_Roman-Empire_trans'] = make_config(GCN_BASE,
    dataset='Roman-Empire', task='vertex', setting='transductive',
    make_undirected=True,
    num_seeds=10, num_splits=1,
)

# Actor - paper: GHC=36.40
EXPERIMENTS['GHC_Actor_trans'] = make_config(GHC_BASE,
    dataset='Actor', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)

# Chameleon - paper: GHC=74.78
EXPERIMENTS['GHC_Chameleon_trans'] = make_config(GHC_BASE,
    dataset='Chameleon', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True, make_undirected=True,
    num_seeds=10, num_splits=10,
)

# Squirrel - paper: GHC=62.90
EXPERIMENTS['GHC_Squirrel_trans'] = make_config(GHC_BASE,
    dataset='Squirrel', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.5,
    mean_agg=True, root_conn=True, residual=False,
    trans_input=False, trans_output=True,
    add_self_loop=True, make_undirected=True,
    num_seeds=10, num_splits=10,
)

# Minesweeper - paper: GHC=87.49
EXPERIMENTS['GHC_Minesweeper_trans'] = make_config(GHC_BASE,
    dataset='Minesweeper', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.3, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=True,
    trans_input=True, trans_output=False,
    add_self_loop=False, make_undirected=True,
    num_seeds=10, num_splits=1,
)

# === Table 1/2: Inductive ===
EXPERIMENTS['GHC_Cora_ind'] = make_config(EXPERIMENTS['GHC_Cora_trans'],
    setting='inductive',
)
EXPERIMENTS['GHC_Roman-Empire_ind'] = make_config(EXPERIMENTS['GHC_Roman-Empire_trans'],
    setting='inductive',
)

# === Table 3: Graph-level tasks ===
EXPERIMENTS['GHC_ZINC'] = make_config(GHC_BASE,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, mix_dim=64, num_blocks=4,
    dropout=0.0, mix_dropout=0.0,
    mean_agg=True, root_conn=True, residual=True,
    trans_input=False, trans_output=True, add_self_loop=True,
    lr=0.001, weight_decay=0.0, epochs=500, patience=100,
    batch_size=128, num_seeds=3, num_splits=1, use_scheduler=True,
)
EXPERIMENTS['GCN_ZINC'] = make_config(GCN_BASE,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, num_layers=4, dropout=0.0,
    lr=0.001, weight_decay=0.0, epochs=500, patience=100,
    batch_size=128, num_seeds=3, num_splits=1, use_scheduler=True,
)
EXPERIMENTS['MLP_ZINC'] = make_config(MLP_BASE,
    dataset='ZINC', task='graph_regression', setting='inductive',
    hidden_dim=256, num_layers=4, dropout=0.0,
    lr=0.001, weight_decay=0.0, epochs=500, patience=100,
    batch_size=128, num_seeds=3, num_splits=1,
)

# === GHM experiments ===
EXPERIMENTS['GHM_Cora_trans'] = make_config(GHM_BASE,
    dataset='Cora', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=64, dropout=0.6,
    root_conn=True, residual=False, k_hop=2,
    trans_input=False, trans_output=True, add_self_loop=True,
    num_seeds=10, num_splits=10,
)
EXPERIMENTS['GHM_Roman-Empire_trans'] = make_config(GHM_BASE,
    dataset='Roman-Empire', task='vertex', setting='transductive',
    hidden_dim=256, mix_dim=32, dropout=0.3, mix_dropout=0.1,
    root_conn=True, residual=True, k_hop=2,
    trans_input=True, trans_output=False,
    add_self_loop=False, make_undirected=True, normalize_input=True,
    num_seeds=10, num_splits=1,
)


def run_and_save(name, config, force=False):
    """Run experiment and save results."""
    result_file = os.path.join(RESULTS_DIR, f'{name}.json')
    
    if os.path.exists(result_file) and not force:
        print(f"  {name}: already done, skipping")
        with open(result_file) as f:
            d = json.load(f)
        task = config.get('task', 'vertex')
        metric = 'MAE' if 'regression' in task else 'Acc'
        print(f"    -> {d['test_mean']:.4f} ± {d['test_std']:.4f} ({metric})")
        return d
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = run_experiment(config, device=device)
    dt = time.time() - t0
    
    save_data = {
        'name': name,
        'test_mean': result['test_mean'],
        'test_std': result['test_std'],
        'val_mean': result['val_mean'],
        'val_std': result['val_std'],
        'time': dt,
        'config': config,
        'all_test': [r['test'] for r in result['all_results']],
        'all_val': [r['val'] for r in result['all_results']],
    }
    
    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    task = config.get('task', 'vertex')
    metric = 'MAE' if 'regression' in task else 'Acc'
    print(f"\n  {name}: {result['test_mean']:.4f} ± {result['test_std']:.4f} ({metric}) [{dt:.1f}s]")
    return save_data


def print_summary():
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    results = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                d = json.load(f)
            results[d['name']] = d
    
    # Paper reference values
    paper = {
        'GHC_Cora_trans': 78.85, 'GHC_CiteSeer_trans': 66.82, 'GHC_PubMed_trans': 76.31,
        'GHC_Photo_trans': 91.63, 'GHC_Computers_trans': 82.12,
        'GHC_Roman-Empire_trans': 92.27, 'GHC_Actor_trans': 36.40,
        'GHC_Chameleon_trans': 74.78, 'GHC_Squirrel_trans': 62.90,
        'GHC_Minesweeper_trans': 87.49,
        'GHC_Cora_ind': 76.69, 'GHC_Roman-Empire_ind': 86.23,
        'GCN_Cora_trans': 78.43, 'GCN_Roman-Empire_trans': 82.72,
        'GHC_ZINC': 0.337, 'GCN_ZINC': 0.426,
        'GHM_Cora_trans': 77.29, 'GHM_Roman-Empire_trans': 72.08,
    }
    
    print(f"\n{'Name':<35s} {'Ours':>15s} {'Paper':>10s} {'Diff':>8s}")
    print("-" * 70)
    for name, d in sorted(results.items()):
        task = d['config'].get('task', 'vertex')
        is_regression = 'regression' in task
        ours = d['test_mean']
        if not is_regression:
            ours_str = f"{ours*100:.2f} ± {d['test_std']*100:.2f}"
        else:
            ours_str = f"{ours:.3f} ± {d['test_std']:.3f}"
        
        if name in paper:
            pval = paper[name]
            if not is_regression:
                diff = ours*100 - pval
            else:
                diff = ours - pval
            diff_str = f"{diff:+.2f}"
            pval_str = f"{pval:.2f}"
        else:
            diff_str = ""
            pval_str = ""
        
        print(f"{name:<35s} {ours_str:>15s} {pval_str:>10s} {diff_str:>8s}")


# Priority order
PRIORITY = [
    'GHC_Cora_trans',
    'GHC_Roman-Empire_trans',
    'GHC_CiteSeer_trans',
    'GHC_PubMed_trans',
    'GHC_Photo_trans',
    'GHC_Computers_trans',
    'GHC_Actor_trans',
    'GHC_Chameleon_trans',
    'GHC_Squirrel_trans',
    'GHC_Minesweeper_trans',
    'GCN_Cora_trans',
    'GCN_Roman-Empire_trans',
    'MLP_Cora_trans',
    'GHC_Cora_ind',
    'GHC_Roman-Empire_ind',
    'GHC_ZINC',
    'GCN_ZINC',
    'MLP_ZINC',
    'GHM_Cora_trans',
    'GHM_Roman-Empire_trans',
]


if __name__ == '__main__':
    force = '--force' in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        print_summary()
        sys.exit(0)
    
    # Select experiments
    if len(sys.argv) > 1 and sys.argv[1] not in ['--force', 'summary']:
        exp_names = [a for a in sys.argv[1:] if a != '--force']
    else:
        exp_names = PRIORITY
    
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"Unknown: {name}")
            continue
        try:
            run_and_save(name, EXPERIMENTS[name], force=force)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()
    
    print_summary()
