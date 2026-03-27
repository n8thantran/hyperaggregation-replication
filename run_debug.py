"""
Debug script to investigate performance gaps, especially on Roman-Empire.
"""
import sys
import json
import torch
import numpy as np
import time

sys.path.insert(0, '/workspace')
from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_roman_empire():
    """Test various configs for Roman-Empire"""
    base = {
        'dataset': 'Roman-Empire', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
        'hidden_dim': 256, 'mix_dim': 32, 'dropout': 0.3, 'input_dropout': 0.0,
        'mix_dropout': 0.1, 'num_blocks': 4, 'mean_agg': False, 'root_conn': True,
        'residual': True, 'trans_input': True, 'trans_output': False,
        'add_self_loop': False, 'make_undirected': True, 'normalize_input': True,
        'lr': 0.001, 'weight_decay': 5e-4, 'epochs': 500, 'patience': 100,
        'num_seeds': 2, 'num_splits': 2,
    }
    
    configs = {
        'base': {},
        'input_act': {'input_activation': True},
        'blocks_6': {'num_blocks': 6},
        'mix_dim_64': {'mix_dim': 64},
    }
    
    results = {}
    for name, overrides in configs.items():
        config = {**base, **overrides}
        print(f"\n=== Roman-Empire {name}: {overrides if overrides else 'base'} ===")
        try:
            result = run_experiment(config, device=device)
            acc = result['test_mean'] * 100
            std = result['test_std'] * 100
            results[name] = f"{acc:.2f}±{std:.2f}"
            print(f"  → {acc:.2f}±{std:.2f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = f"FAILED: {e}"
    
    print("\n\n=== Summary ===")
    for name, res in results.items():
        print(f"  {name}: {res}")

if __name__ == '__main__':
    test_roman_empire()
