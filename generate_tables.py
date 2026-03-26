#!/usr/bin/env python3
"""Generate result tables from saved JSON results."""
import json
import os
import glob

results_dir = 'results'

def load_result(filename):
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def fmt(mean, std, pct=True):
    if pct:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    else:
        return f"{mean:.3f} ± {std:.3f}"

print("=" * 80)
print("TABLE 1: Homophilic Datasets (Transductive Vertex Classification)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20} {'Paper GCN':<20} {'Our GCN':<20}")
print("-" * 95)

table1_datasets = [
    ('Cora', '78.85 ± 2.14', '78.43 ± 1.35'),
    ('CiteSeer', '66.82 ± 1.66', '66.75 ± 1.89'),
    ('PubMed', '76.31 ± 2.71', '75.62 ± 2.28'),
    ('Computers', '82.12 ± 1.91', '80.72 ± 1.79'),
    ('Photo', '91.63 ± 0.79', '91.16 ± 0.83'),
]

for ds, paper_ghc, paper_gcn in table1_datasets:
    ghc = load_result(f'GHC_{ds}_trans.json')
    gcn = load_result(f'GCN_{ds}_trans.json')
    
    our_ghc = fmt(ghc['test_mean'], ghc['test_std']) if ghc else 'N/A'
    our_gcn = fmt(gcn['test_mean'], gcn['test_std']) if gcn else 'N/A'
    
    print(f"{ds:<15} {paper_ghc:<20} {our_ghc:<20} {paper_gcn:<20} {our_gcn:<20}")

print()
print("=" * 80)
print("TABLE 2: Heterophilic Datasets (Transductive Vertex Classification)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20}")
print("-" * 55)

table2_datasets = [
    ('Chameleon', '74.78 ± 1.82'),
    ('Squirrel', '62.90 ± 1.47'),
    ('Actor', '36.40 ± 1.46'),
    ('Minesweeper', '87.49 ± 0.61'),
    ('RomanEmpire', '92.27 ± 0.57'),
]

for ds, paper_ghc in table2_datasets:
    ghc = load_result(f'GHC_{ds}_trans.json')
    our_ghc = fmt(ghc['test_mean'], ghc['test_std']) if ghc else 'N/A'
    print(f"{ds:<15} {paper_ghc:<20} {our_ghc:<20}")

print()
print("=" * 80)
print("TABLE 3: Graph-Level Regression (ZINC)")
print("=" * 80)
print(f"{'Dataset':<15} {'Paper GHC':<20} {'Our GHC':<20}")
print("-" * 55)

zinc = load_result('GHC_ZINC_graph.json')
if zinc:
    our_zinc = fmt(zinc['test_mean'], zinc['test_std'], pct=False)
else:
    our_zinc = 'N/A'
print(f"{'ZINC':<15} {'0.337 ± 0.020':<20} {our_zinc:<20}")

print()
print("=" * 80)
print("BASELINES")
print("=" * 80)
for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    name = os.path.basename(f).replace('.json', '')
    r = load_result(os.path.basename(f))
    if r and 'test_mean' in r:
        print(f"  {name}: {r['test_mean']:.4f} ± {r.get('test_std', 0):.4f}")

# Also write markdown version
with open('results/results_tables.md', 'w') as out:
    out.write("# Replication Results\n\n")
    
    out.write("## Table 1: Homophilic Datasets (Transductive)\n\n")
    out.write("| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |\n")
    out.write("|---------|-----------|---------|-----------|----------|\n")
    for ds, paper_ghc, paper_gcn in table1_datasets:
        ghc = load_result(f'GHC_{ds}_trans.json')
        gcn = load_result(f'GCN_{ds}_trans.json')
        our_ghc = fmt(ghc['test_mean'], ghc['test_std']) if ghc else 'N/A'
        our_gcn = fmt(gcn['test_mean'], gcn['test_std']) if gcn else 'N/A'
        out.write(f"| {ds} | {paper_ghc} | {our_ghc} | {paper_gcn} | {our_gcn} |\n")
    
    out.write("\n## Table 2: Heterophilic Datasets (Transductive)\n\n")
    out.write("| Dataset | Paper GHC | Our GHC |\n")
    out.write("|---------|-----------|----------|\n")
    for ds, paper_ghc in table2_datasets:
        ghc = load_result(f'GHC_{ds}_trans.json')
        our_ghc = fmt(ghc['test_mean'], ghc['test_std']) if ghc else 'N/A'
        out.write(f"| {ds} | {paper_ghc} | {our_ghc} |\n")
    
    out.write("\n## Table 3: Graph-Level (ZINC)\n\n")
    out.write("| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |\n")
    out.write("|---------|------------------|------------------|\n")
    zinc = load_result('GHC_ZINC_graph.json')
    our_zinc = fmt(zinc['test_mean'], zinc['test_std'], pct=False) if zinc else 'N/A'
    out.write(f"| ZINC | 0.337 ± 0.020 | {our_zinc} |\n")
    
    out.write("\n## All Results\n\n")
    for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        name = os.path.basename(f).replace('.json', '')
        r = load_result(os.path.basename(f))
        if r and 'test_mean' in r:
            out.write(f"- **{name}**: {r['test_mean']:.4f} ± {r.get('test_std', 0):.4f}\n")

print("\nSaved markdown tables to results/results_tables.md")
