#!/usr/bin/env python3
"""Generate result tables from saved JSON results."""

import json
import os

RESULTS_DIR = 'results'

# Paper reference values
PAPER_VALUES = {
    # Table 1: Homophilic (accuracy %)
    'GHC_Cora': (78.85, 2.14),
    'GHC_CiteSeer': (66.82, 1.66),
    'GHC_PubMed': (76.31, 2.71),
    'GHC_Computers': (82.12, 1.91),
    'GHC_Photo': (91.63, 0.79),
    # Table 2: Heterophilic (accuracy %)
    'GHC_Chameleon': (74.78, 1.82),
    'GHC_Squirrel': (62.90, 1.47),
    'GHC_Actor': (36.40, 1.46),
    'GHC_Minesweeper': (87.49, 0.61),
    'GHC_RomanEmpire': (92.27, 0.57),
    # Table 3: ZINC (MAE, lower is better)
    'GHC_ZINC': (0.337, 0.020),
    # Baselines
    'GCN_Cora': (78.43, 1.36),
    'GCN_CiteSeer': (66.75, 1.42),
    'GCN_PubMed': (75.62, 2.45),
    'MLP_Cora': (56.29, 1.82),
    'MLP_Chameleon': (45.57, 2.07),
    'GCN_Chameleon': (69.63, 1.73),
}


def load_result(name, suffix='trans'):
    """Load result from JSON file."""
    fname = os.path.join(RESULTS_DIR, f'{name}_{suffix}.json')
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)
    return None


def fmt_acc(result, scale=100):
    """Format accuracy result."""
    if result is None:
        return '-'
    mean = result['test_mean'] * scale
    std = result['test_std'] * scale
    return f"{mean:.2f}±{std:.2f}"


def fmt_mae(result):
    """Format MAE result."""
    if result is None:
        return '-'
    return f"{result['test_mean']:.3f}±{result['test_std']:.3f}"


def fmt_paper(key):
    """Format paper reference value."""
    if key not in PAPER_VALUES:
        return '-'
    mean, std = PAPER_VALUES[key]
    if key == 'GHC_ZINC':
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean:.2f}±{std:.2f}"


def main():
    lines = []
    lines.append("# HyperAggregation Replication Results\n")
    lines.append("Replication of 'HyperAggregation: Aggregating over Graph Edges with Hypernetworks'\n")
    
    # ================================================================
    # Table 1: Homophilic datasets
    # ================================================================
    lines.append("## Table 1: Homophilic Datasets (Transductive Vertex Classification)\n")
    lines.append("| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |")
    lines.append("|---------|-----------|---------|-----------|---------|")
    
    for dataset in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']:
        ghc = load_result(f'GHC_{dataset}')
        gcn = load_result(f'GCN_{dataset}')
        lines.append(f"| {dataset} | {fmt_paper(f'GHC_{dataset}')} | {fmt_acc(ghc)} | {fmt_paper(f'GCN_{dataset}')} | {fmt_acc(gcn)} |")
    
    # ================================================================
    # Table 2: Heterophilic datasets
    # ================================================================
    lines.append("\n## Table 2: Heterophilic Datasets (Transductive Vertex Classification)\n")
    lines.append("| Dataset | Paper GHC | Our GHC |")
    lines.append("|---------|-----------|---------|")
    
    dataset_map = {
        'Chameleon': 'Chameleon',
        'Squirrel': 'Squirrel', 
        'Actor': 'Actor',
        'Minesweeper': 'Minesweeper',
        'Roman-Empire': 'RomanEmpire',
    }
    
    for display_name, file_name in dataset_map.items():
        ghc = load_result(f'GHC_{file_name}')
        paper_key = f'GHC_{file_name}'
        lines.append(f"| {display_name} | {fmt_paper(paper_key)} | {fmt_acc(ghc)} |")
    
    # ================================================================
    # Table 3: Graph-level (ZINC)
    # ================================================================
    lines.append("\n## Table 3: Graph-Level Regression (ZINC)\n")
    lines.append("| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |")
    lines.append("|---------|------------------|-----------------|")
    
    zinc = load_result('GHC_ZINC', suffix='graph')
    lines.append(f"| ZINC | {fmt_paper('GHC_ZINC')} | {fmt_mae(zinc)} |")
    
    # ================================================================
    # Baselines
    # ================================================================
    lines.append("\n## Baselines\n")
    lines.append("| Dataset | Model | Paper | Ours |")
    lines.append("|---------|-------|-------|------|")
    
    baselines = [
        ('Cora', 'GCN'), ('Cora', 'MLP'),
        ('CiteSeer', 'GCN'), ('PubMed', 'GCN'),
        ('Chameleon', 'GCN'), ('Chameleon', 'MLP'),
    ]
    
    for dataset, model in baselines:
        key = f'{model}_{dataset}'
        result = load_result(key)
        lines.append(f"| {dataset} | {model} | {fmt_paper(key)} | {fmt_acc(result)} |")
    
    # ================================================================
    # Table 5: Ablation Study
    # ================================================================
    ablation_file = os.path.join(RESULTS_DIR, 'ablation_table5.json')
    if os.path.exists(ablation_file):
        with open(ablation_file) as f:
            ablation = json.load(f)
        
        lines.append("\n## Table 5: Ablation Study\n")
        lines.append("| Hyperparameter | Cora Δ (ours) | Cora Δ (paper) | RE Δ (ours) | RE Δ (paper) |")
        lines.append("|----------------|---------------|----------------|-------------|--------------|")
        
        cora_base = ablation.get('cora_base', 0)
        re_base = ablation.get('re_base', 0)
        lines.append(f"| Base | {cora_base:.2f} | 78.85 | {re_base:.2f} | 92.27 |")
        
        ablation_names = [
            ("Self-loops", -2.84, -0.13),
            ("Normalize input", -0.66, -0.01),
            ("Residual", -3.09, -1.22),
            ("Root connection", -0.50, -1.72),
            ("Mean aggregate", -1.35, -2.64),
            ("Trans HA input", 0.47, -1.15),
            ("Trans HA output", -4.83, -0.08),
        ]
        
        for name, paper_cd, paper_rd in ablation_names:
            cora_data = ablation.get(f'cora_{name}', {})
            re_data = ablation.get(f're_{name}', {})
            cd = cora_data.get('delta', 0) if isinstance(cora_data, dict) else 0
            rd = re_data.get('delta', 0) if isinstance(re_data, dict) else 0
            lines.append(f"| {name} | {cd:+.2f} | {paper_cd:+.2f} | {rd:+.2f} | {paper_rd:+.2f} |")
    
    # Write output
    table_text = '\n'.join(lines)
    
    output_file = os.path.join(RESULTS_DIR, 'results_tables.md')
    with open(output_file, 'w') as f:
        f.write(table_text)
    
    print(table_text)
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
