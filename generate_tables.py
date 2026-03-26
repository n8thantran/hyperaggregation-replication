#!/usr/bin/env python3
"""Generate result tables from saved JSON files."""
import json
import os
import numpy as np

results_dir = 'results'

def load_result(filename):
    """Load a result JSON file, return (mean, std) or None."""
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d.get('test_mean'), d.get('test_std')
    return None, None

def fmt(mean, std, multiply=100, decimals=2):
    """Format mean ± std."""
    if mean is None:
        return "N/A"
    if multiply:
        mean *= multiply
        std *= multiply
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"

print("=" * 80)
print("TABLE 1: Homophilic Datasets (Transductive, Accuracy %)")
print("=" * 80)

homo_datasets = ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
paper_ghc_homo = {
    'Cora': (78.85, 2.14), 'CiteSeer': (66.82, 1.66), 'PubMed': (76.31, 2.71),
    'Computers': (82.12, 1.91), 'Photo': (91.63, 0.79)
}

print(f"{'Dataset':<12} {'Paper GHC':<18} {'My GHC':<18}")
print("-" * 50)
for ds in homo_datasets:
    pm, ps = paper_ghc_homo[ds]
    m, s = load_result(f'GHC_{ds}_trans.json')
    paper_str = f"{pm:.2f}±{ps:.2f}"
    my_str = fmt(m, s)
    print(f"{ds:<12} {paper_str:<18} {my_str:<18}")

print()
print("=" * 80)
print("TABLE 2: Heterophilic Datasets (Transductive, Accuracy %)")  
print("=" * 80)

hetero_datasets = ['Chameleon', 'Squirrel', 'Actor', 'Minesweeper', 'RomanEmpire']
paper_ghc_hetero = {
    'Chameleon': (74.78, 1.82), 'Squirrel': (62.90, 1.47), 'Actor': (36.40, 1.46),
    'Minesweeper': (87.49, 0.61), 'RomanEmpire': (92.27, 0.57)
}

print(f"{'Dataset':<14} {'Paper GHC':<18} {'My GHC':<18}")
print("-" * 50)
for ds in hetero_datasets:
    pm, ps = paper_ghc_hetero[ds]
    m, s = load_result(f'GHC_{ds}_trans.json')
    paper_str = f"{pm:.2f}±{ps:.2f}"
    my_str = fmt(m, s)
    print(f"{ds:<14} {paper_str:<18} {my_str:<18}")

print()
print("=" * 80)
print("TABLE 3: Graph-Level Tasks")
print("=" * 80)

m, s = load_result('GHC_ZINC_graph.json')
if m is not None:
    print(f"ZINC GHC (MAE): Paper = 0.337±0.020, Mine = {fmt(m, s, multiply=None, decimals=3)}")
else:
    print("ZINC result not yet available")

print()
print("=" * 80)
print("BASELINES")
print("=" * 80)

baselines = [
    ('GCN_Cora_trans.json', 'GCN Cora', (78.43, 0.85)),
    ('GCN_CiteSeer_trans.json', 'GCN CiteSeer', (66.75, 1.86)),
    ('GCN_PubMed_trans.json', 'GCN PubMed', (75.62, 2.24)),
    ('GCN_Chameleon_trans.json', 'GCN Chameleon', (69.63, 2.41)),
    ('MLP_Cora_trans.json', 'MLP Cora', (56.29, 2.08)),
    ('MLP_Chameleon_trans.json', 'MLP Chameleon', (45.57, 1.77)),
]

print(f"{'Model/Dataset':<20} {'Paper':<18} {'Mine':<18}")
print("-" * 56)
for fname, label, (pm, ps) in baselines:
    m, s = load_result(fname)
    paper_str = f"{pm:.2f}±{ps:.2f}"
    my_str = fmt(m, s)
    print(f"{label:<20} {paper_str:<18} {my_str:<18}")

# Save markdown version
with open('results/results_tables.md', 'w') as f:
    f.write("# Replication Results\n\n")
    
    f.write("## Table 1: Homophilic Datasets (Transductive, Accuracy %)\n\n")
    f.write("| Dataset | Paper GHC | My GHC |\n")
    f.write("|---------|-----------|--------|\n")
    for ds in homo_datasets:
        pm, ps = paper_ghc_homo[ds]
        m, s = load_result(f'GHC_{ds}_trans.json')
        f.write(f"| {ds} | {pm:.2f}±{ps:.2f} | {fmt(m, s)} |\n")
    
    f.write("\n## Table 2: Heterophilic Datasets (Transductive, Accuracy %)\n\n")
    f.write("| Dataset | Paper GHC | My GHC |\n")
    f.write("|---------|-----------|--------|\n")
    for ds in hetero_datasets:
        pm, ps = paper_ghc_hetero[ds]
        m, s = load_result(f'GHC_{ds}_trans.json')
        f.write(f"| {ds} | {pm:.2f}±{ps:.2f} | {fmt(m, s)} |\n")
    
    f.write("\n## Table 3: Graph-Level (ZINC MAE, lower is better)\n\n")
    m, s = load_result('GHC_ZINC_graph.json')
    if m is not None:
        f.write(f"| ZINC | 0.337±0.020 | {fmt(m, s, multiply=None, decimals=3)} |\n")
    else:
        f.write("| ZINC | 0.337±0.020 | Not yet available |\n")
    
    f.write("\n## Baselines\n\n")
    f.write("| Model/Dataset | Paper | Mine |\n")
    f.write("|--------------|-------|------|\n")
    for fname, label, (pm, ps) in baselines:
        m, s = load_result(fname)
        f.write(f"| {label} | {pm:.2f}±{ps:.2f} | {fmt(m, s)} |\n")

print("\nSaved results/results_tables.md")
