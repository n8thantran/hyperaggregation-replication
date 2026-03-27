"""Focused sweep on Roman-Empire to close the gap."""
import json, os, time
from train import run_experiment

# Paper's base config for Roman-Empire (inferred from ablation table):
# add_self_loop=False, make_undirected=True, normalize_input=True, residual=True
# root_conn=True, mean_agg=False, trans_input=True, trans_output=False
# hidden_dim=256, mix_dim=32, mix_dropout=0.1, dropout=0.3

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
    'epochs': 1000,
    'patience': 200,
    'num_seeds': 3,
    'num_splits': 3,
}

configs = []

# Try different number of blocks
for n_blocks in [2, 3, 4, 5, 6]:
    c = base.copy()
    c['num_blocks'] = n_blocks
    c['name'] = f'blocks_{n_blocks}'
    configs.append(c)

# Try different weight decays
for wd in [0.0, 1e-4, 1e-3]:
    c = base.copy()
    c['num_blocks'] = 2
    c['weight_decay'] = wd
    c['name'] = f'wd_{wd}'
    configs.append(c)

# Try input_activation
c = base.copy()
c['num_blocks'] = 2
c['input_activation'] = True
c['name'] = 'input_act'
configs.append(c)

# Try with scheduler
c = base.copy()
c['num_blocks'] = 2
c['use_scheduler'] = True
c['name'] = 'scheduler'
configs.append(c)

# Try different learning rates
for lr in [0.0005, 0.002, 0.005]:
    c = base.copy()
    c['num_blocks'] = 2
    c['lr'] = lr
    c['name'] = f'lr_{lr}'
    configs.append(c)

results = {}
for cfg in configs:
    name = cfg.pop('name')
    print(f"\n=== {name} ===")
    t0 = time.time()
    result = run_experiment(cfg, device='cuda')
    dt = time.time() - t0
    print(f"  Test: {result['test_mean']*100:.2f} ± {result['test_std']*100:.2f} ({dt:.0f}s)")
    results[name] = {
        'test_mean': result['test_mean'],
        'test_std': result['test_std'],
    }

print("\n=== SUMMARY ===")
for name, r in sorted(results.items(), key=lambda x: -x[1]['test_mean']):
    print(f"  {name}: {r['test_mean']*100:.2f} ± {r['test_std']*100:.2f}")

with open('results/roman_empire_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
