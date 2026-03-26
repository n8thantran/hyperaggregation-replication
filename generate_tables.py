#!/usr/bin/env python3
"""Generate result tables from saved JSON results."""

import json
import os

results_dir = 'results'

def load_result(filename):
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def fmt(result, metric='accuracy'):
    """Format result as mean ± std."""
    if result is None:
        return "N/A"
    if metric == 'accuracy':
        return f"{result['test_mean']*100:.2f} ± {result['test_std']*100:.2f}"
    elif metric == 'mae':
        return f"{result['test_mean']:.4f} ± {result['test_std']:.4f}"

# Table 1: Homophilic datasets (transductive)
print("=" * 80)
print("Table 1: Homophilic Datasets (Transductive)")
print("=" * 80)

paper_ghc = {
    'Cora': '78.85 ± 2.14',
    'CiteSeer': '66.82 ± 1.66',
    'PubMed': '76.31 ± 2.71',
    'Computers': '82.12 ± 1.91',
    'Photo': '91.63 ± 0.79',
}

paper_gcn = {
    'Cora': '78.43 ± 2.09',
    'CiteSeer': '66.75 ± 1.55',
    'PubMed': '75.62 ± 2.56',
    'Computers': '82.44 ± 1.76',
    'Photo': '91.40 ± 1.30',
}

datasets_homo = ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20} {'Paper GCN':<20} {'Our GCN':<20}")
print("-" * 80)
for ds in datasets_homo:
    ghc = load_result(f'GHC_{ds}_trans.json')
    gcn = load_result(f'GCN_{ds}_trans.json')
    print(f"{ds:<15} {paper_ghc.get(ds, 'N/A'):<20} {fmt(ghc):<20} {paper_gcn.get(ds, 'N/A'):<20} {fmt(gcn) if gcn else 'N/A':<20}")

# Table 2: Heterophilic datasets (transductive)
print("\n" + "=" * 80)
print("Table 2: Heterophilic Datasets (Transductive)")
print("=" * 80)

paper_ghc_het = {
    'Chameleon': '74.78 ± 1.82',
    'Squirrel': '62.90 ± 1.47',
    'Actor': '36.40 ± 1.46',
    'Minesweeper': '87.49 ± 0.61',
    'RomanEmpire': '92.27 ± 0.57',
}

datasets_het = ['Chameleon', 'Squirrel', 'Actor', 'Minesweeper', 'RomanEmpire']
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20}")
print("-" * 55)
for ds in datasets_het:
    ghc = load_result(f'GHC_{ds}_trans.json')
    print(f"{ds:<15} {paper_ghc_het.get(ds, 'N/A'):<20} {fmt(ghc):<20}")

# Table 3: Graph-level (ZINC)
print("\n" + "=" * 80)
print("Table 3: Graph-Level Tasks (ZINC MAE ↓)")
print("=" * 80)
zinc = load_result('GHC_ZINC_graph.json')
print(f"{'Model':<15} {'Paper':<20} {'Ours':<20}")
print("-" * 55)
print(f"{'GHC':<15} {'0.337 ± 0.020':<20} {fmt(zinc, 'mae') if zinc else 'N/A':<20}")

# Table 5: Ablation
print("\n" + "=" * 80)
print("Table 5: Ablation Study (GHC)")
print("=" * 80)
abl_path = os.path.join(results_dir, 'ablation_table5.json')
if os.path.exists(abl_path):
    with open(abl_path) as f:
        abl = json.load(f)
    
    cora_base = abl.get('cora_base', 0)
    re_base = abl.get('re_base', 0)
    
    print(f"{'Hyperparameter':<25} {'Cora Δ':>10} {'Paper Δ':>10} {'RE Δ':>10} {'Paper Δ':>10}")
    print("-" * 70)
    print(f"{'Base':<25} {cora_base:>10.2f} {'78.85':>10} {re_base:>10.2f} {'92.27':>10}")
    
    paper_deltas = {
        'Self-loops': (-2.84, -0.13),
        'Normalize input': (-0.66, -0.01),
        'Residual': (-3.09, -1.22),
        'Root connection': (-0.50, -1.72),
        'Mean aggregate': (-1.35, -2.64),
        'Trans HA input': (0.47, -1.15),
        'Trans HA output': (-4.83, -0.08),
    }
    
    for name, (pcd, prd) in paper_deltas.items():
        cora_data = abl.get(f'cora_{name}', {})
        re_data = abl.get(f're_{name}', {})
        cd = cora_data.get('delta', 'N/A') if isinstance(cora_data, dict) else 'N/A'
        rd = re_data.get('delta', 'N/A') if isinstance(re_data, dict) else 'N/A'
        cd_str = f"{cd:+.2f}" if isinstance(cd, (int, float)) else cd
        rd_str = f"{rd:+.2f}" if isinstance(rd, (int, float)) else rd
        print(f"{name:<25} {cd_str:>10} {pcd:>+10.2f} {rd_str:>10} {prd:>+10.2f}")
else:
    print("Ablation results not yet available.")

# Save as markdown
with open('results/results_tables.md', 'w') as f:
    f.write("# Replication Results\n\n")
    
    f.write("## Table 1: Homophilic Datasets (Transductive)\n\n")
    f.write("| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |\n")
    f.write("|---------|-----------|---------|-----------|--------|\n")
    for ds in datasets_homo:
        ghc = load_result(f'GHC_{ds}_trans.json')
        gcn = load_result(f'GCN_{ds}_trans.json')
        f.write(f"| {ds} | {paper_ghc.get(ds, 'N/A')} | {fmt(ghc)} | {paper_gcn.get(ds, 'N/A')} | {fmt(gcn) if gcn else 'N/A'} |\n")
    
    f.write("\n## Table 2: Heterophilic Datasets (Transductive)\n\n")
    f.write("| Dataset | Paper GHC | Our GHC |\n")
    f.write("|---------|-----------|--------|\n")
    for ds in datasets_het:
        ghc = load_result(f'GHC_{ds}_trans.json')
        f.write(f"| {ds} | {paper_ghc_het.get(ds, 'N/A')} | {fmt(ghc)} |\n")
    
    f.write("\n## Table 3: Graph-Level (ZINC MAE ↓)\n\n")
    f.write("| Model | Paper | Ours |\n")
    f.write("|-------|-------|------|\n")
    zinc = load_result('GHC_ZINC_graph.json')
    f.write(f"| GHC | 0.337 ± 0.020 | {fmt(zinc, 'mae') if zinc else 'N/A'} |\n")
    
    f.write("\n## Table 5: Ablation Study\n\n")
    if os.path.exists(abl_path):
        f.write("| Hyperparameter | Cora Δ | Paper Δ | RE Δ | Paper Δ |\n")
        f.write("|----------------|--------|---------|------|--------|\n")
        f.write(f"| Base | {cora_base:.2f} | 78.85 | {re_base:.2f} | 92.27 |\n")
        for name, (pcd, prd) in paper_deltas.items():
            cora_data = abl.get(f'cora_{name}', {})
            re_data = abl.get(f're_{name}', {})
            cd = cora_data.get('delta', 'N/A') if isinstance(cora_data, dict) else 'N/A'
            rd = re_data.get('delta', 'N/A') if isinstance(re_data, dict) else 'N/A'
            cd_str = f"{cd:+.2f}" if isinstance(cd, (int, float)) else cd
            rd_str = f"{rd:+.2f}" if isinstance(rd, (int, float)) else rd
            f.write(f"| {name} | {cd_str} | {pcd:+.2f} | {rd_str} | {prd:+.2f} |\n")

print("\n\nResults saved to results/results_tables.md")
