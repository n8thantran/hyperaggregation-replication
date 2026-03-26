#!/usr/bin/env python3
"""Quick hyperparameter sweep on Chameleon - minimal."""

import torch
from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base = {
    'dataset': 'Chameleon', 'model': 'GHC', 'task': 'vertex', 'setting': 'transductive',
    'input_dropout': 0.0, 'mix_dropout': 0.0, 'root_conn': True,
    'add_self_loop': True, 'make_undirected': True, 'input_activation': False,
    'lr': 0.001, 'weight_decay': 0, 'epochs': 300, 'patience': 50,
    'num_seeds': 1, 'num_splits': 3,
    'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 2,
    'dropout': 0.5, 'trans_input': False, 'trans_output': True,
    'residual': False, 'normalize_input': False, 'mean_agg': True,
}

variations = [
    ('baseline', {}),
    ('residual', {'residual': True}),
    ('res+norm', {'residual': True, 'normalize_input': True}),
    ('res+4blk', {'residual': True, 'num_blocks': 4}),
    ('res+d0.3', {'residual': True, 'dropout': 0.3}),
    ('not_undir', {'make_undirected': False}),
]

for name, delta in variations:
    cfg = {**base, **delta}
    result = run_experiment(cfg, device=device)
    print(f"{name:15s}: val={result['val_mean']:.4f} test={result['test_mean']:.4f}")
