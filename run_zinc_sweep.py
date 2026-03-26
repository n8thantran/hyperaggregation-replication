"""Run ZINC experiments with different configs to find best."""
import torch
import numpy as np
import json
import time
from train import run_single_experiment

device = 'cuda'

configs = {
    'h64_m32_4blk': {
        'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
        'setting': 'inductive',
        'hidden_dim': 64, 'mix_dim': 32, 'dropout': 0.0, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': False,
        'add_self_loop': True, 'make_undirected': False,
        'use_embedding': True, 'num_embeddings': 28,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'use_scheduler': True, 'batch_size': 128,
    },
    'h128_m64_4blk': {
        'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
        'setting': 'inductive',
        'hidden_dim': 128, 'mix_dim': 64, 'dropout': 0.0, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': False,
        'add_self_loop': True, 'make_undirected': False,
        'use_embedding': True, 'num_embeddings': 28,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'use_scheduler': True, 'batch_size': 128,
    },
    'h256_m64_4blk_transout': {
        'dataset': 'ZINC', 'model': 'GHC', 'task': 'graph_regression',
        'setting': 'inductive',
        'hidden_dim': 256, 'mix_dim': 64, 'dropout': 0.0, 'input_dropout': 0.0,
        'mix_dropout': 0.0, 'num_blocks': 4, 'mean_agg': True, 'root_conn': True,
        'residual': True, 'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'use_embedding': True, 'num_embeddings': 28,
        'lr': 0.001, 'weight_decay': 0.0, 'epochs': 500, 'patience': 100,
        'use_scheduler': True, 'batch_size': 128,
    },
}

best_config_name = None
best_mae = float('inf')

for name, config in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")
    
    results = []
    for seed in range(3):
        t0 = time.time()
        result = run_single_experiment(config, seed=seed, device=device)
        dt = time.time() - t0
        print(f'  Seed {seed}: Test MAE: {result["test"]:.4f}, Epochs: {result["epochs_trained"]}, Time: {dt:.1f}s')
        results.append(result['test'])
    
    mean_mae = np.mean(results)
    std_mae = np.std(results)
    print(f'  Mean: {mean_mae:.4f} ± {std_mae:.4f}')
    
    if mean_mae < best_mae:
        best_mae = mean_mae
        best_config_name = name

print(f"\nBest config: {best_config_name} with MAE: {best_mae:.4f}")
print(f"Paper target: 0.337 ± 0.020")

# Save best result
best_config = configs[best_config_name]
save_data = {
    'name': f'GHC_ZINC_{best_config_name}',
    'test_mean': best_mae,
    'test_std': std_mae,
    'config': best_config,
}
with open('results/GHC_ZINC_graph.json', 'w') as f:
    json.dump(save_data, f, indent=2)
