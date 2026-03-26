#!/usr/bin/env python3
"""Generate result tables comparing our results with paper."""
import json
import os
import numpy as np

results_dir = 'results'

def load_result(filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# Table 1: Homophilic datasets (transductive)
print("=" * 80)
print("TABLE 1: Homophilic Datasets (Transductive Vertex Classification)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20} {'Paper GCN':<20} {'Our GCN':<20}")
print("-" * 95)

homo_datasets = [
    ('Cora', 'GHC_Cora_trans.json', '78.85±2.14', 'GCN_Cora_trans.json', '78.43±1.32'),
    ('CiteSeer', 'GHC_CiteSeer_trans.json', '66.82±1.66', 'GCN_CiteSeer_trans.json', '66.75±1.86'),
    ('PubMed', 'GHC_PubMed_trans.json', '76.31±2.71', 'GCN_PubMed_trans.json', '75.62±1.97'),
    ('Computers', 'GHC_Computers_trans.json', '82.12±1.91', None, '81.37±1.70'),
    ('Photo', 'GHC_Photo_trans.json', '91.63±0.79', None, '90.36±0.83'),
]

for name, ghc_file, paper_ghc, gcn_file, paper_gcn in homo_datasets:
    ghc_result = load_result(ghc_file)
    gcn_result = load_result(gcn_file) if gcn_file else None
    
    ghc_str = f"{ghc_result['test_mean']*100:.2f}±{ghc_result['test_std']*100:.2f}" if ghc_result else "N/A"
    gcn_str = f"{gcn_result['test_mean']*100:.2f}±{gcn_result['test_std']*100:.2f}" if gcn_result else "N/A"
    
    print(f"{name:<15} {paper_ghc:<20} {ghc_str:<20} {paper_gcn:<20} {gcn_str:<20}")

print()

# Table 2: Heterophilic datasets (transductive)
print("=" * 80)
print("TABLE 2: Heterophilic Datasets (Transductive Vertex Classification)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20}")
print("-" * 55)

hetero_datasets = [
    ('Chameleon', 'GHC_Chameleon_trans.json', '74.78±1.82'),
    ('Squirrel', 'GHC_Squirrel_trans.json', '62.90±1.47'),
    ('Actor', 'GHC_Actor_trans.json', '36.40±1.46'),
    ('Minesweeper', 'GHC_Minesweeper_trans.json', '87.49±0.61'),
    ('Roman-Empire', 'GHC_RomanEmpire_trans.json', '92.27±0.57'),
]

for name, ghc_file, paper_ghc in hetero_datasets:
    ghc_result = load_result(ghc_file)
    ghc_str = f"{ghc_result['test_mean']*100:.2f}±{ghc_result['test_std']*100:.2f}" if ghc_result else "N/A"
    print(f"{name:<15} {paper_ghc:<20} {ghc_str:<20}")

print()

# Table 3: Graph-level tasks
print("=" * 80)
print("TABLE 3: Graph-Level Tasks (ZINC - MAE, lower is better)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20}")
print("-" * 55)

zinc_result = load_result('GHC_ZINC_graph.json')
if zinc_result:
    zinc_str = f"{zinc_result['test_mean']:.3f}±{zinc_result['test_std']:.3f}"
    print(f"{'ZINC':<15} {'0.337±0.020':<20} {zinc_str:<20}")

print()

# Baselines
print("=" * 80)
print("BASELINE COMPARISONS")
print("=" * 80)
print(f"{'Dataset':<15} {'Model':<10} {'Paper':<20} {'Ours':<20}")
print("-" * 65)

baselines = [
    ('Cora', 'GCN', 'GCN_Cora_trans.json', '78.43±1.32'),
    ('Cora', 'MLP', 'MLP_Cora_trans.json', '56.29±1.54'),
    ('CiteSeer', 'GCN', 'GCN_CiteSeer_trans.json', '66.75±1.86'),
    ('PubMed', 'GCN', 'GCN_PubMed_trans.json', '75.62±1.97'),
    ('Chameleon', 'GCN', 'GCN_Chameleon_trans.json', '69.63±2.03'),
    ('Chameleon', 'MLP', 'MLP_Chameleon_trans.json', '45.57±4.04'),
]

for name, model, file, paper in baselines:
    result = load_result(file)
    our_str = f"{result['test_mean']*100:.2f}±{result['test_std']*100:.2f}" if result else "N/A"
    print(f"{name:<15} {model:<10} {paper:<20} {our_str:<20}")

# Save as markdown
with open('results/results_tables.md', 'w') as f:
    f.write("# Replication Results\n\n")
    f.write("## Table 1: Homophilic Datasets (Transductive)\n\n")
    f.write("| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |\n")
    f.write("|---------|-----------|---------|-----------|--------|\n")
    for name, ghc_file, paper_ghc, gcn_file, paper_gcn in homo_datasets:
        ghc_result = load_result(ghc_file)
        gcn_result = load_result(gcn_file) if gcn_file else None
        ghc_str = f"{ghc_result['test_mean']*100:.2f}±{ghc_result['test_std']*100:.2f}" if ghc_result else "N/A"
        gcn_str = f"{gcn_result['test_mean']*100:.2f}±{gcn_result['test_std']*100:.2f}" if gcn_result else "N/A"
        f.write(f"| {name} | {paper_ghc} | {ghc_str} | {paper_gcn} | {gcn_str} |\n")
    
    f.write("\n## Table 2: Heterophilic Datasets (Transductive)\n\n")
    f.write("| Dataset | Paper GHC | Our GHC |\n")
    f.write("|---------|-----------|--------|\n")
    for name, ghc_file, paper_ghc in hetero_datasets:
        ghc_result = load_result(ghc_file)
        ghc_str = f"{ghc_result['test_mean']*100:.2f}±{ghc_result['test_std']*100:.2f}" if ghc_result else "N/A"
        f.write(f"| {name} | {paper_ghc} | {ghc_str} |\n")
    
    f.write("\n## Table 3: Graph-Level (ZINC MAE, lower is better)\n\n")
    f.write("| Dataset | Paper GHC | Our GHC |\n")
    f.write("|---------|-----------|--------|\n")
    if zinc_result:
        zinc_str = f"{zinc_result['test_mean']:.3f}±{zinc_result['test_std']:.3f}"
        f.write(f"| ZINC | 0.337±0.020 | {zinc_str} |\n")
    
    f.write("\n## Baselines\n\n")
    f.write("| Dataset | Model | Paper | Ours |\n")
    f.write("|---------|-------|-------|------|\n")
    for name, model, file, paper in baselines:
        result = load_result(file)
        our_str = f"{result['test_mean']*100:.2f}±{result['test_std']*100:.2f}" if result else "N/A"
        f.write(f"| {name} | {model} | {paper} | {our_str} |\n")

print("\nResults saved to results/results_tables.md")
