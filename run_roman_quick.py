"""Quick sweep on Roman-Empire: 1 split, 1 seed per config."""
import json, os, time
from train import run_experiment

base = {
    'dataset': 'Roman-Empire',
    'model': 'GHC',
    'task': 'vertex',
    'setting': 'transductive',
    'hidden_dim': 256,
    'mix_dim': 32,
    'dropout': 0.3,
    'input_dropout': 0.0,
    'mix_dropout': 0.1,
    'mean_agg': False,
    'root_conn': True,
    'residual': True,
    'trans_input': True,
    'trans_output': False,
    'add_self_loop': False,
    'make_undirected': True,
    'normalize_input': True,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 500,
    'patience': 100,
    'num_seeds': 1,
    'num_splits': 1,
}

configs = {}

# Different depths
for n_blocks in [2, 3, 4, 5, 6, 8]:
    c = base.copy()
    c['num_blocks'] = n_blocks
    configs[f'blocks_{n_blocks}'] = c

# Input dropout
for idp in [0.1, 0.3, 0.5]:
    c = base.copy()
    c['num_blocks'] = 4
    c['input_dropout'] = idp
    configs[f'idrop_{idp}'] = c

# Try different input_activation
c = base.copy()
c['num_blocks'] = 4
c['input_activation'] = True
configs['input_act'] = c

results = {}
for name, cfg in configs.items():
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_experiment(cfg, device='cuda')
    dt = time.time() - t0
    print(f"  Test: {result['test_mean']*100:.2f} ({dt:.0f}s)")
    results[name] = result['test_mean']

print("\n=== SUMMARY ===")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name}: {acc*100:.2f}")

with open('results/roman_quick_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
