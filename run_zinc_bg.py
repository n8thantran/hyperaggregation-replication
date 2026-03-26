#!/usr/bin/env python3
"""Run ZINC experiment with 3 seeds, 300 epochs each. Save results incrementally."""
import torch
import json
import os
import time
import numpy as np
from train import run_single_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

config = {
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
    'epochs': 300,
    'patience': 50,
    'batch_size': 128,
    'use_scheduler': True,
}

all_results = []
for seed in range(3):
    t0 = time.time()
    result = run_single_experiment(config, seed=seed, device=device)
    dt = time.time() - t0
    all_results.append(result)
    print(f'Seed {seed}: Test MAE={result["test"]:.4f}, Epochs={result["epochs_trained"]}, Time={dt:.1f}s')
    
    # Save intermediate
    test_maes = [r['test'] for r in all_results]
    save_data = {
        'test_mean': float(np.mean(test_maes)),
        'test_std': float(np.std(test_maes)) if len(test_maes) > 1 else 0.0,
        'seeds_done': len(all_results),
        'all_test': test_maes,
        'config': config,
    }
    with open('results/GHC_ZINC_graph.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'  Saved intermediate results ({len(all_results)}/3 seeds)')

test_maes = [r['test'] for r in all_results]
print(f'\nFinal ZINC GHC: MAE = {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}')
print(f'Paper target: 0.337 ± 0.020')
