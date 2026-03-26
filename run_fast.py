"""
Run experiments quickly - fewer seeds/splits for development.
Then run full for final results.
"""
import torch
import numpy as np
import json
import os
import sys
import time
import traceback

sys.path.insert(0, '/workspace')
from train import run_experiment

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_config(**kwargs):
    return kwargs

# Paper target values
PAPER = {
    'GHC_Cora_trans': 78.85, 'GHC_CiteSeer_trans': 66.82, 'GHC_PubMed_trans': 76.31,
    'GHC_Photo_trans': 91.63, 'GHC_Computers_trans': 82.12,
    'GHC_Roman-Empire_trans': 92.27, 'GHC_Actor_trans': 36.40,
    'GHC_Chameleon_trans': 74.78, 'GHC_Squirrel_trans': 62.90,
    'GHC_Minesweeper_trans': 87.49,
    'GHC_Cora_ind': 76.69, 'GHC_Roman-Empire_ind': 86.23,
    'GCN_Cora_trans': 78.43, 'GCN_Roman-Empire_trans': 82.72,
    'MLP_Cora_trans': 56.29,
    'GHC_ZINC': 0.337, 'GCN_ZINC': 0.426, 'MLP_ZINC': 0.731,
    'GHM_Cora_trans': 77.33, 'GHM_Roman-Empire_trans': 83.97,
}

# All experiment configs - optimized per dataset
EXPERIMENTS = {
    # === TABLE 1: Homophilic transductive ===
    'GHC_Cora_trans': {
        'dataset': 'Cora', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_CiteSeer_trans': {
        'dataset': 'CiteSeer', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_PubMed_trans': {
        'dataset': 'PubMed', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_Computers_trans': {
        'dataset': 'Computers', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_Photo_trans': {
        'dataset': 'Photo', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    
    # === TABLE 2: Heterophilic transductive ===
    'GHC_Chameleon_trans': {
        'dataset': 'Chameleon', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': True, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_Squirrel_trans': {
        'dataset': 'Squirrel', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': True, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_Actor_trans': {
        'dataset': 'Actor', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GHC_Minesweeper_trans': {
        'dataset': 'Minesweeper', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.3, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': True, 'trans_output': False, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': True,
        'add_self_loop': False, 'make_undirected': True, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_Roman-Empire_trans': {
        'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4,
        'dropout': 0.3, 'input_dropout': 0.0, 'mix_dropout': 0.1,
        'trans_input': True, 'trans_output': False, 'input_activation': False,
        'mean_agg': False, 'root_conn': True, 'residual': True,
        'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 1,
    },
    
    # === Baselines ===
    'GCN_Cora_trans': {
        'dataset': 'Cora', 'model': 'GCN', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'input_dropout': 0.0,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'MLP_Cora_trans': {
        'dataset': 'Cora', 'model': 'MLP', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'input_dropout': 0.0,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    'GCN_Roman-Empire_trans': {
        'dataset': 'Roman-Empire', 'model': 'GCN', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'input_dropout': 0.0,
        'make_undirected': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 1,
    },
    
    # === Inductive ===
    'GHC_Cora_ind': {
        'dataset': 'Cora', 'model': 'GHC', 'task': 'vertex', 'setting': 'inductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': False,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
    
    # === Graph-level (ZINC) ===
    'GHC_ZINC': {
        'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression', 'setting': 'inductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4,
        'dropout': 0.0, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'mean_agg': True, 'root_conn': True, 'residual': True,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'batch_size': 128, 'num_seeds': 3, 'num_splits': 1, 'use_scheduler': True,
    },
    'GCN_ZINC': {
        'dataset': 'ZINC', 'model': 'GCN', 'task': 'graph_regression', 'setting': 'inductive',
        'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.0, 'input_dropout': 0.0,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'batch_size': 128, 'num_seeds': 3, 'num_splits': 1, 'use_scheduler': True,
    },
    'MLP_ZINC': {
        'dataset': 'ZINC', 'model': 'MLP', 'task': 'graph_regression', 'setting': 'inductive',
        'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.0, 'input_dropout': 0.0,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'batch_size': 128, 'num_seeds': 3, 'num_splits': 1,
    },
    
    # === GHM ===
    'GHM_Cora_trans': {
        'dataset': 'Cora', 'model': 'GHM', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
        'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
        'trans_input': False, 'trans_output': True, 'input_activation': False,
        'root_conn': True, 'residual': False, 'k_hop': 2,
        'add_self_loop': True, 'make_undirected': False, 'normalize_input': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 10,
    },
}


def run_and_save(name, config, force=False):
    """Run experiment and save results."""
    result_file = os.path.join(RESULTS_DIR, f'{name}.json')
    
    if os.path.exists(result_file) and not force:
        with open(result_file) as f:
            d = json.load(f)
        task = config.get('task', 'vertex')
        is_reg = 'regression' in task
        if is_reg:
            print(f"  {name}: {d['test_mean']:.3f} ± {d['test_std']:.3f} (cached)")
        else:
            print(f"  {name}: {d['test_mean']*100:.2f} ± {d['test_std']*100:.2f} (cached)")
        return d
    
    print(f"\nRunning: {name}")
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
    is_reg = 'regression' in task
    if is_reg:
        print(f"  {name}: {result['test_mean']:.3f} ± {result['test_std']:.3f} [{dt:.0f}s]")
    else:
        print(f"  {name}: {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f} [{dt:.0f}s]")
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
        
        if name in PAPER:
            pval = PAPER[name]
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


if __name__ == '__main__':
    force = '--force' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--force']
    
    if args and args[0] == 'summary':
        print_summary()
        sys.exit(0)
    
    if args and args[0] == 'all':
        exp_names = list(EXPERIMENTS.keys())
    elif args:
        exp_names = args
    else:
        # Default priority
        exp_names = [
            'GHC_Cora_trans', 'GHC_CiteSeer_trans', 'GHC_PubMed_trans',
            'GHC_Computers_trans', 'GHC_Photo_trans',
            'GHC_Chameleon_trans', 'GHC_Squirrel_trans', 'GHC_Actor_trans',
            'GHC_Minesweeper_trans', 'GHC_Roman-Empire_trans',
            'GCN_Cora_trans', 'MLP_Cora_trans', 'GCN_Roman-Empire_trans',
            'GHC_Cora_ind',
            'GHC_ZINC', 'GCN_ZINC', 'MLP_ZINC',
            'GHM_Cora_trans',
        ]
    
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
