#!/usr/bin/env python3
"""
Reproduce key results from "HyperAggregation: Aggregating over Graph Edges with Hypernetworks"

This script runs the main experiments from Tables 1, 2, and 3 of the paper.
Results are saved to /workspace/results/ as JSON files and a summary table.

Usage:
    python run_reproduce.py              # Run all experiments
    python run_reproduce.py --quick      # Quick mode (fewer seeds, shorter training)
    python run_reproduce.py --table 1    # Run only Table 1 experiments
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

from train import run_experiment

# ============================================================================
# Experiment configurations from the paper
# ============================================================================

# Table 1: Homophilic datasets (transductive vertex classification)
TABLE1_CONFIGS = {
    'GHC_Cora': {
        'dataset': 'Cora', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': False, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_CiteSeer': {
        'dataset': 'CiteSeer', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': False, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_PubMed': {
        'dataset': 'PubMed', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': False, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_Computers': {
        'dataset': 'Computers', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': False, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
    'GHC_Photo': {
        'dataset': 'Photo', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.6, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': False, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
}

# Table 2: Heterophilic datasets (transductive vertex classification)
TABLE2_CONFIGS = {
    'GHC_Chameleon': {
        'dataset': 'Chameleon', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_Squirrel': {
        'dataset': 'Squirrel', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 5, 'num_splits': 1,
    },
    'GHC_Actor': {
        'dataset': 'Actor', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 10, 'num_splits': 1,
    },
    'GHC_Minesweeper': {
        'dataset': 'Minesweeper', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 2, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 5, 'num_splits': 1,
    },
    'GHC_RomanEmpire': {
        'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
        'residual': True, 'trans_input': True, 'trans_output': False,
        'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 5, 'num_splits': 1,
    },
}

# Table 3: Graph-level (ZINC)
TABLE3_CONFIGS = {
    'GHC_ZINC': {
        'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
        'setting': 'inductive',
        'hidden_dim': 64, 'mix_dim': 32, 'dropout': 0.0, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': False,
        'add_self_loop': True, 'make_undirected': False,
        'use_embedding': True, 'num_embeddings': 28,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'use_scheduler': True, 'batch_size': 128,
        'num_seeds': 3, 'num_splits': 1,
    },
}

# Baseline configs
BASELINE_CONFIGS = {
    'GCN_Cora': {
        'dataset': 'Cora', 'model': 'GCN', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
    'MLP_Cora': {
        'dataset': 'Cora', 'model': 'MLP', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
    'GCN_CiteSeer': {
        'dataset': 'CiteSeer', 'model': 'GCN', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
    'GCN_PubMed': {
        'dataset': 'PubMed', 'model': 'GCN', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200, 'patience': 50,
        'num_seeds': 5, 'num_splits': 1,
    },
}


def run_and_save(name, config, results_dir='results', skip_existing=False):
    """Run experiment and save results."""
    # Determine filename
    model = config['model']
    dataset = config['dataset'].replace('-', '')
    task = config.get('task', 'vertex')
    setting = config.get('setting', 'transductive')
    suffix = 'graph' if task in ['graph', 'graph_regression'] else 'trans'
    filename = f"{model}_{dataset}_{suffix}.json"
    filepath = os.path.join(results_dir, filename)
    
    if skip_existing and os.path.exists(filepath):
        print(f"  Skipping {name} (already exists)")
        with open(filepath) as f:
            return json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    result = run_experiment(config, device=device)
    elapsed = time.time() - t0
    
    # Format for saving
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
    
    os.makedirs(results_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    if is_regression:
        print(f"  Result: {metric_name} = {result['test_mean']:.4f} ± {result['test_std']:.4f}")
    else:
        print(f"  Result: {metric_name} = {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f}")
    
    return save_data


def generate_summary_table(results_dir='results'):
    """Generate summary table from saved results."""
    lines = []
    lines.append("# HyperAggregation Replication Results\n")
    
    # Paper reference values
    paper_values = {
        'GHC_Cora': ('78.85±2.14', 'Acc'),
        'GHC_CiteSeer': ('66.82±1.66', 'Acc'),
        'GHC_PubMed': ('76.31±2.71', 'Acc'),
        'GHC_Computers': ('82.12±1.91', 'Acc'),
        'GHC_Photo': ('91.63±0.79', 'Acc'),
        'GHC_Chameleon': ('74.78±1.82', 'Acc'),
        'GHC_Squirrel': ('62.90±1.47', 'Acc'),
        'GHC_Actor': ('36.40±1.46', 'Acc'),
        'GHC_Minesweeper': ('87.49±0.61', 'Acc'),
        'GHC_RomanEmpire': ('92.27±0.57', 'Acc'),
        'GHC_ZINC': ('0.337±0.020', 'MAE'),
        'GCN_Cora': ('78.43±1.36', 'Acc'),
        'GCN_CiteSeer': ('66.75±1.42', 'Acc'),
        'GCN_PubMed': ('75.62±2.45', 'Acc'),
        'MLP_Cora': ('56.29±1.82', 'Acc'),
    }
    
    # Table 1: Homophilic
    lines.append("## Table 1: Homophilic Datasets (Transductive)\n")
    lines.append("| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |")
    lines.append("|---------|-----------|---------|-----------|---------|")
    
    for dataset in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']:
        ghc_key = f'GHC_{dataset}'
        gcn_key = f'GCN_{dataset}'
        
        # Our GHC
        ghc_file = os.path.join(results_dir, f'GHC_{dataset}_trans.json')
        if os.path.exists(ghc_file):
            with open(ghc_file) as f:
                d = json.load(f)
            our_ghc = f"{d['test_mean']*100:.2f}±{d['test_std']*100:.2f}"
        else:
            our_ghc = '-'
        
        # Our GCN
        gcn_file = os.path.join(results_dir, f'GCN_{dataset}_trans.json')
        if os.path.exists(gcn_file):
            with open(gcn_file) as f:
                d = json.load(f)
            our_gcn = f"{d['test_mean']*100:.2f}±{d['test_std']*100:.2f}"
        else:
            our_gcn = '-'
        
        paper_ghc = paper_values.get(ghc_key, ('-', ''))[0]
        paper_gcn = paper_values.get(gcn_key, ('-', ''))[0]
        
        lines.append(f"| {dataset} | {paper_ghc} | {our_ghc} | {paper_gcn} | {our_gcn} |")
    
    # Table 2: Heterophilic
    lines.append("\n## Table 2: Heterophilic Datasets (Transductive)\n")
    lines.append("| Dataset | Paper GHC | Our GHC |")
    lines.append("|---------|-----------|---------|")
    
    for dataset in ['Chameleon', 'Squirrel', 'Actor', 'Minesweeper', 'Roman-Empire']:
        ghc_key = f'GHC_{dataset.replace("-", "")}'
        fname = dataset.replace('-', '')
        
        ghc_file = os.path.join(results_dir, f'GHC_{fname}_trans.json')
        if os.path.exists(ghc_file):
            with open(ghc_file) as f:
                d = json.load(f)
            our_ghc = f"{d['test_mean']*100:.2f}±{d['test_std']*100:.2f}"
        else:
            our_ghc = '-'
        
        paper_ghc = paper_values.get(ghc_key, ('-', ''))[0]
        lines.append(f"| {dataset} | {paper_ghc} | {our_ghc} |")
    
    # Table 3: Graph-level
    lines.append("\n## Table 3: Graph-Level (ZINC)\n")
    lines.append("| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |")
    lines.append("|---------|------------------|-----------------|")
    
    zinc_file = os.path.join(results_dir, 'GHC_ZINC_graph.json')
    if os.path.exists(zinc_file):
        with open(zinc_file) as f:
            d = json.load(f)
        if 'test_std' in d and d['test_std'] > 0:
            our_zinc = f"{d['test_mean']:.3f}±{d['test_std']:.3f}"
        else:
            our_zinc = f"{d['test_mean']:.3f}"
    else:
        our_zinc = '-'
    
    lines.append(f"| ZINC | 0.337±0.020 | {our_zinc} |")
    
    # Baselines
    lines.append("\n## Baselines\n")
    lines.append("| Dataset | Model | Paper | Ours |")
    lines.append("|---------|-------|-------|------|")
    
    for key in ['GCN_Cora', 'MLP_Cora', 'GCN_CiteSeer', 'GCN_PubMed']:
        model, dataset = key.split('_', 1)
        fname = os.path.join(results_dir, f'{key}_trans.json')
        if os.path.exists(fname):
            with open(fname) as f:
                d = json.load(f)
            ours = f"{d['test_mean']*100:.2f}±{d['test_std']*100:.2f}"
        else:
            ours = '-'
        paper = paper_values.get(key, ('-', ''))[0]
        lines.append(f"| {dataset} | {model} | {paper} | {ours} |")
    
    table_text = '\n'.join(lines)
    
    with open(os.path.join(results_dir, 'results_tables.md'), 'w') as f:
        f.write(table_text)
    
    print("\n" + table_text)
    return table_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer seeds)')
    parser.add_argument('--table', type=int, default=0, help='Run only specific table (1, 2, or 3)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip experiments with existing results')
    parser.add_argument('--baselines', action='store_true', help='Also run baselines')
    args = parser.parse_args()
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect configs to run
    configs = {}
    
    if args.table == 0 or args.table == 1:
        configs.update(TABLE1_CONFIGS)
    if args.table == 0 or args.table == 2:
        configs.update(TABLE2_CONFIGS)
    if args.table == 0 or args.table == 3:
        configs.update(TABLE3_CONFIGS)
    if args.baselines or args.table == 0:
        configs.update(BASELINE_CONFIGS)
    
    # Quick mode: reduce seeds
    if args.quick:
        for name, cfg in configs.items():
            cfg['num_seeds'] = min(cfg.get('num_seeds', 10), 3)
            if cfg.get('task') == 'graph_regression':
                cfg['epochs'] = min(cfg.get('epochs', 500), 300)
    
    # Run experiments
    all_results = {}
    for name, config in configs.items():
        try:
            result = run_and_save(name, config, results_dir, 
                                  skip_existing=args.skip_existing)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary table
    generate_summary_table(results_dir)
    
    print(f"\nAll results saved to {results_dir}/")


if __name__ == '__main__':
    main()
