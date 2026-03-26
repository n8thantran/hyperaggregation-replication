#!/usr/bin/env python3
"""
Run ZINC graph regression experiment.
Paper target: GHC achieves 0.337 ± 0.020 MAE on ZINC.
"""

import torch
import json
import os
import time
from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

os.makedirs('results', exist_ok=True)

# ZINC GHC configuration
zinc_config = {
    'dataset': 'zinc',
    'model': 'GHC',
    'task': 'graph_regression',
    'setting': 'inductive',
    'hidden_dim': 256,
    'mix_dim': 64,
    'dropout': 0.0,
    'input_dropout': 0.0,
    'mix_dropout': 0.0,
    'num_blocks': 4,
    'mean_agg': True,
    'root_conn': True,
    'residual': True,
    'trans_input': False,
    'trans_output': True,
    'add_self_loop': True,
    'make_undirected': False,
    'normalize_input': False,
    'use_embedding': True,
    'num_embeddings': 28,
    'lr': 0.001,
    'weight_decay': 0.0,
    'epochs': 200,
    'patience': 50,
    'batch_size': 128,
    'num_seeds': 3,
    'num_splits': 1,
    'use_scheduler': True,
}

print("=" * 60)
print("Running ZINC GHC experiment (200 epochs, 3 seeds)")
print("=" * 60)

t0 = time.time()
result = run_experiment(zinc_config, device=device)
dt = time.time() - t0

print(f"\nZINC GHC: MAE = {result['test_mean']:.4f} ± {result['test_std']:.4f}")
print(f"Paper target: 0.337 ± 0.020")
print(f"Time: {dt:.1f}s")

# Save result
result_data = {
    'test_mean': result['test_mean'],
    'test_std': result['test_std'],
    'val_mean': result['val_mean'],
    'val_std': result['val_std'],
    'all_results': result['all_results'],
    'config': zinc_config,
    'paper_target': '0.337 ± 0.020',
}
with open('results/GHC_ZINC_graph.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print("\nSaved to results/GHC_ZINC_graph.json")
