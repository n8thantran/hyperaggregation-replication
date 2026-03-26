#!/usr/bin/env python3
"""Re-run Roman-Empire with correct hyperparameters from paper ablation table."""
import torch
import json
import os
import numpy as np
from train import run_experiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

config = {
    'name': 'GHC_RomanEmpire_trans',
    'dataset': 'Roman-Empire',
    'model': 'GHC',
    'setting': 'transductive',
    'task': 'vertex',
    'hidden_dim': 256,
    'mix_dim': 32,          # paper says 32 for Roman-Empire
    'num_blocks': 4,
    'dropout': 0.3,          # paper says 0.3
    'input_dropout': 0.0,
    'mix_dropout': 0.1,      # paper says 0.1
    'mean_agg': False,        # ablation: "yes" gives -2.64
    'root_conn': True,
    'residual': True,
    'trans_input': True,      # ablation: "no" gives -1.15
    'trans_output': False,    # ablation: "yes" gives -0.08
    'add_self_loop': False,   # ablation: "yes" gives -0.13
    'make_undirected': True,  # ablation: "no" gives -5.89
    'normalize_input': True,  # ablation: "no" gives -0.01
    'input_activation': False,
    'lr': 0.001,
    'weight_decay': 0,
    'epochs': 500,
    'patience': 50,
    'num_seeds': 5,
    'num_splits': 1,
}

print(f"Running Roman-Empire with corrected config...")
print(f"Key settings: mix_dim={config['mix_dim']}, dropout={config['dropout']}, mix_dropout={config['mix_dropout']}")
print(f"  mean_agg={config['mean_agg']}, trans_input={config['trans_input']}, trans_output={config['trans_output']}")
print(f"  add_self_loop={config['add_self_loop']}, make_undirected={config['make_undirected']}")

result = run_experiment(config, device=device)
print(f"\nRoman-Empire GHC: {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f}")
print(f"Paper target: 92.27 ± 0.57")

# Save
save_data = {
    'test_mean': result['test_mean'],
    'test_std': result['test_std'],
    'val_mean': result['val_mean'],
    'val_std': result['val_std'],
    'all_results': result['all_results'],
    'config': config,
}
with open('results/GHC_RomanEmpire_trans.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print("Saved to results/GHC_RomanEmpire_trans.json")
