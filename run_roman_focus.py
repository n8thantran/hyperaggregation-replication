"""Focused Roman-Empire sweep around input_dropout=0.5."""
import json, time
from train import run_experiment

base = {
    'dataset': 'Roman-Empire',
    'model': 'GHC',
    'task': 'vertex',
    'setting': 'transductive',
    'hidden_dim': 256,
    'mix_dim': 32,
    'dropout': 0.3,
    'input_dropout': 0.5,
    'mix_dropout': 0.1,
    'num_blocks': 4,
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

# Different input dropouts around 0.5
for idp in [0.4, 0.5, 0.6, 0.7, 0.8]:
    c = base.copy()
    c['input_dropout'] = idp
    configs[f'idrop_{idp}'] = c

# Depth with input_dropout
for nb in [4, 5, 6, 8]:
    c = base.copy()
    c['num_blocks'] = nb
    c['input_dropout'] = 0.5
    configs[f'depth_{nb}_idrop_0.5'] = c

# Vary model dropout with input_dropout=0.5
for dp in [0.1, 0.2, 0.3, 0.5]:
    c = base.copy()
    c['input_dropout'] = 0.5
    c['dropout'] = dp
    configs[f'dp_{dp}_idrop_0.5'] = c

# Higher mix_dim with input_dropout
for md in [32, 64]:
    c = base.copy()
    c['input_dropout'] = 0.5
    c['mix_dim'] = md
    configs[f'mix_{md}_idrop_0.5'] = c

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

with open('results/roman_focus_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
