"""
Run all experiments for HyperAggregation paper replication.
Covers Tables 1, 2, 3 from the paper.
"""

import torch
import json
import os
import time
import numpy as np
import sys
import gc

from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================

# Common defaults
BASE_VERTEX = {
    'task': 'vertex',
    'epochs': 500,
    'patience': 100,
    'num_seeds': 10,
    'num_splits': 1,
}

BASE_GRAPH = {
    'epochs': 500,
    'patience': 100,
    'num_seeds': 3,
    'num_splits': 1,
}

# ---- TABLE 1: Homophilic datasets (transductive) ----

GHC_Cora_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Cora_trans',
    'dataset': 'Cora', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_CiteSeer_trans = {
    **BASE_VERTEX,
    'name': 'GHC_CiteSeer_trans',
    'dataset': 'CiteSeer', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.6, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_PubMed_trans = {
    **BASE_VERTEX,
    'name': 'GHC_PubMed_trans',
    'dataset': 'PubMed', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_Computers_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Computers_trans',
    'dataset': 'Computers', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_Photo_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Photo_trans',
    'dataset': 'Photo', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

# ---- TABLE 2: Heterophilic datasets (transductive) ----

GHC_Chameleon_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Chameleon_trans',
    'dataset': 'Chameleon', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_Squirrel_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Squirrel_trans',
    'dataset': 'Squirrel', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_Actor_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Actor_trans',
    'dataset': 'Actor', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 5e-4,
}

GHC_Minesweeper_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Minesweeper_trans',
    'dataset': 'Minesweeper', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.3, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': False,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 0.0,
}

GHC_RomanEmpire_trans = {
    **BASE_VERTEX,
    'name': 'GHC_Roman-Empire_trans',
    'dataset': 'Roman-Empire', 'model': 'GHC', 'setting': 'transductive',
    'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4,
    'dropout': 0.3, 'input_dropout': 0.0, 'mix_dropout': 0.1,
    'mean_agg': False, 'root_conn': True, 'residual': True,
    'trans_input': True, 'trans_output': False,
    'add_self_loop': False, 'make_undirected': True,
    'normalize_input': True,
    'lr': 0.001, 'weight_decay': 0.0,
}

# ---- Baselines: GCN and MLP ----

GCN_Cora_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Cora_trans',
    'dataset': 'Cora', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

MLP_Cora = {
    **BASE_VERTEX,
    'name': 'MLP_Cora',
    'dataset': 'Cora', 'model': 'MLP', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_CiteSeer_trans = {
    **BASE_VERTEX,
    'name': 'GCN_CiteSeer_trans',
    'dataset': 'CiteSeer', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

MLP_CiteSeer = {
    **BASE_VERTEX,
    'name': 'MLP_CiteSeer',
    'dataset': 'CiteSeer', 'model': 'MLP', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_PubMed_trans = {
    **BASE_VERTEX,
    'name': 'GCN_PubMed_trans',
    'dataset': 'PubMed', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

MLP_PubMed = {
    **BASE_VERTEX,
    'name': 'MLP_PubMed',
    'dataset': 'PubMed', 'model': 'MLP', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_RomanEmpire_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Roman-Empire_trans',
    'dataset': 'Roman-Empire', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 0.0,
}

MLP_RomanEmpire = {
    **BASE_VERTEX,
    'name': 'MLP_Roman-Empire',
    'dataset': 'Roman-Empire', 'model': 'MLP', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 0.0,
}

GCN_Chameleon_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Chameleon_trans',
    'dataset': 'Chameleon', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_Squirrel_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Squirrel_trans',
    'dataset': 'Squirrel', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_Actor_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Actor_trans',
    'dataset': 'Actor', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_Minesweeper_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Minesweeper_trans',
    'dataset': 'Minesweeper', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 0.0,
}

GCN_Computers_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Computers_trans',
    'dataset': 'Computers', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

GCN_Photo_trans = {
    **BASE_VERTEX,
    'name': 'GCN_Photo_trans',
    'dataset': 'Photo', 'model': 'GCN', 'setting': 'transductive',
    'hidden_dim': 256, 'num_layers': 2,
    'dropout': 0.5, 'input_dropout': 0.0,
    'lr': 0.01, 'weight_decay': 5e-4,
}

# ---- TABLE 3: Graph-level ----

GHC_ZINC = {
    **BASE_GRAPH,
    'name': 'GHC_ZINC',
    'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
    'hidden_dim': 128, 'mix_dim': 64, 'num_blocks': 4,
    'dropout': 0.0, 'input_dropout': 0.0, 'mix_dropout': 0.0,
    'mean_agg': True, 'root_conn': True, 'residual': True,
    'trans_input': False, 'trans_output': True,
    'add_self_loop': True, 'make_undirected': False,
    'normalize_input': False,
    'lr': 0.001, 'weight_decay': 0.0,
    'batch_size': 128,
    'use_scheduler': True,
    'epochs': 1000, 'patience': 200,
}

GCN_ZINC = {
    **BASE_GRAPH,
    'name': 'GCN_ZINC',
    'dataset': 'ZINC', 'model': 'GCN', 'task': 'graph_regression',
    'hidden_dim': 128, 'num_layers': 4,
    'dropout': 0.0, 'input_dropout': 0.0,
    'lr': 0.001, 'weight_decay': 0.0,
    'batch_size': 128,
    'use_scheduler': True,
    'epochs': 1000, 'patience': 200,
}

MLP_ZINC = {
    **BASE_GRAPH,
    'name': 'MLP_ZINC',
    'dataset': 'ZINC', 'model': 'MLP', 'task': 'graph_regression',
    'hidden_dim': 128, 'num_layers': 4,
    'dropout': 0.0, 'input_dropout': 0.0,
    'lr': 0.001, 'weight_decay': 0.0,
    'batch_size': 128,
    'use_scheduler': True,
    'epochs': 1000, 'patience': 200,
}

# ============================================================================
# EXPERIMENT GROUPS
# ============================================================================

# Priority order: GHC on key datasets first, then baselines, then extras
ALL_EXPERIMENTS = [
    # Table 1 - GHC homophilic
    GHC_Cora_trans,
    GHC_CiteSeer_trans,
    GHC_PubMed_trans,
    GHC_Computers_trans,
    GHC_Photo_trans,
    # Table 2 - GHC heterophilic
    GHC_Chameleon_trans,
    GHC_Squirrel_trans,
    GHC_Actor_trans,
    GHC_Minesweeper_trans,
    GHC_RomanEmpire_trans,
    # Baselines
    GCN_Cora_trans,
    MLP_Cora,
    GCN_CiteSeer_trans,
    MLP_CiteSeer,
    GCN_PubMed_trans,
    MLP_PubMed,
    GCN_RomanEmpire_trans,
    MLP_RomanEmpire,
    GCN_Chameleon_trans,
    GCN_Squirrel_trans,
    GCN_Actor_trans,
    GCN_Minesweeper_trans,
    GCN_Computers_trans,
    GCN_Photo_trans,
    # Table 3 - Graph-level
    GHC_ZINC,
    GCN_ZINC,
    MLP_ZINC,
]


def run_and_save(config):
    """Run experiment and save results."""
    name = config['name']
    result_path = f'results/{name}.json'
    
    # Skip if already done
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        task = config.get('task', 'vertex')
        if task == 'graph_regression':
            print(f"  SKIP {name}: already done (MAE={existing['test_mean']:.4f} ± {existing['test_std']:.4f})")
        else:
            print(f"  SKIP {name}: already done ({existing['test_mean']*100:.2f} ± {existing['test_std']*100:.2f})")
        return existing
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    try:
        result = run_experiment(config, device=device)
        dt = time.time() - t0
        
        task = config.get('task', 'vertex')
        if task == 'graph_regression':
            print(f"\n>>> {name}: MAE = {result['test_mean']:.4f} ± {result['test_std']:.4f} ({dt:.0f}s)")
        else:
            print(f"\n>>> {name}: Acc = {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f} ({dt:.0f}s)")
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        print(f"\n>>> FAILED {name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # Allow selecting specific experiments from command line
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        experiments = [e for e in ALL_EXPERIMENTS if e['name'] in selected]
        if not experiments:
            print(f"No matching experiments found. Available:")
            for e in ALL_EXPERIMENTS:
                print(f"  {e['name']}")
            sys.exit(1)
    else:
        experiments = ALL_EXPERIMENTS
    
    print(f"Running {len(experiments)} experiments on {device}")
    print(f"Experiments: {[e['name'] for e in experiments]}")
    
    results = {}
    for config in experiments:
        result = run_and_save(config)
        if result is not None:
            results[config['name']] = result
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, result in results.items():
        task = 'graph_regression' if 'ZINC' in name else 'vertex'
        if task == 'graph_regression':
            print(f"  {name:40s}: MAE = {result['test_mean']:.4f} ± {result['test_std']:.4f}")
        else:
            print(f"  {name:40s}: Acc = {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f}")
