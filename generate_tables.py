#!/usr/bin/env python3
"""Generate result tables matching the paper's format."""

import json
import os

results_dir = 'results'

def load_result(filename):
    """Load a result JSON file."""
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def fmt(mean, std, is_mae=False):
    """Format mean±std."""
    if mean is None:
        return "N/A"
    if is_mae:
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean*100:.2f}±{std*100:.2f}"

def generate_all_tables():
    lines = []
    
    # ============================================================
    # Table 1: Homophilic datasets (transductive)
    # ============================================================
    lines.append("# Table 1: Homophilic Datasets (Transductive)")
    lines.append("")
    
    datasets_homo = ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    paper_ghc = {
        'Cora': '78.85±2.14', 'CiteSeer': '66.82±1.66', 'PubMed': '76.31±2.71',
        'Computers': '82.12±1.91', 'Photo': '91.63±0.79'
    }
    paper_gcn = {
        'Cora': '78.43±0.85', 'CiteSeer': '66.75±1.86', 'PubMed': '75.62±2.24',
        'Computers': '83.52±1.73', 'Photo': '90.94±1.06'
    }
    paper_mlp = {
        'Cora': '56.29±2.08', 'CiteSeer': '55.54±3.47', 'PubMed': '68.27±2.29',
        'Computers': '67.43±1.79', 'Photo': '78.75±1.73'
    }
    
    header = f"| {'Model':<15} |"
    for d in datasets_homo:
        header += f" {d:>15} |"
    lines.append(header)
    lines.append("|" + "-"*17 + "|" + (("-"*17 + "|") * len(datasets_homo)))
    
    # Paper GHC
    row = f"| {'Paper GHC':<15} |"
    for d in datasets_homo:
        row += f" {paper_ghc[d]:>15} |"
    lines.append(row)
    
    # Our GHC
    row = f"| {'Our GHC':<15} |"
    for d in datasets_homo:
        r = load_result(f'GHC_{d}_trans.json')
        if r:
            row += f" {fmt(r['test_mean'], r['test_std']):>15} |"
        else:
            row += f" {'N/A':>15} |"
    lines.append(row)
    
    # Paper GCN
    row = f"| {'Paper GCN':<15} |"
    for d in datasets_homo:
        row += f" {paper_gcn[d]:>15} |"
    lines.append(row)
    
    # Our GCN (only some)
    row = f"| {'Our GCN':<15} |"
    for d in datasets_homo:
        r = load_result(f'GCN_{d}_trans.json')
        if r:
            row += f" {fmt(r['test_mean'], r['test_std']):>15} |"
        else:
            row += f" {'N/A':>15} |"
    lines.append(row)
    
    lines.append("")
    
    # ============================================================
    # Table 2: Heterophilic datasets (transductive)
    # ============================================================
    lines.append("# Table 2: Heterophilic Datasets (Transductive)")
    lines.append("")
    
    datasets_hetero = ['Chameleon', 'Squirrel', 'Actor', 'Minesweeper', 'RomanEmpire']
    display_names = {'Chameleon': 'Chameleon', 'Squirrel': 'Squirrel', 'Actor': 'Actor', 
                     'Minesweeper': 'Minesweeper', 'RomanEmpire': 'Roman-Empire'}
    paper_ghc_h = {
        'Chameleon': '74.78±1.82', 'Squirrel': '62.90±1.47', 'Actor': '36.40±1.46',
        'Minesweeper': '87.49±0.61', 'RomanEmpire': '92.27±0.57'
    }
    paper_gcn_h = {
        'Chameleon': '69.63±2.41', 'Squirrel': '59.56±1.92', 'Actor': '33.70±1.26',
        'Minesweeper': '88.72±0.52', 'RomanEmpire': '82.72±0.82'
    }
    
    header = f"| {'Model':<15} |"
    for d in datasets_hetero:
        header += f" {display_names[d]:>15} |"
    lines.append(header)
    lines.append("|" + "-"*17 + "|" + (("-"*17 + "|") * len(datasets_hetero)))
    
    # Paper GHC
    row = f"| {'Paper GHC':<15} |"
    for d in datasets_hetero:
        row += f" {paper_ghc_h[d]:>15} |"
    lines.append(row)
    
    # Our GHC
    row = f"| {'Our GHC':<15} |"
    for d in datasets_hetero:
        r = load_result(f'GHC_{d}_trans.json')
        if r:
            row += f" {fmt(r['test_mean'], r['test_std']):>15} |"
        else:
            row += f" {'N/A':>15} |"
    lines.append(row)
    
    # Paper GCN
    row = f"| {'Paper GCN':<15} |"
    for d in datasets_hetero:
        row += f" {paper_gcn_h[d]:>15} |"
    lines.append(row)
    
    lines.append("")
    
    # ============================================================
    # Table 3: Graph-level (ZINC)
    # ============================================================
    lines.append("# Table 3: Graph-Level Tasks (ZINC MAE ↓)")
    lines.append("")
    
    r = load_result('GHC_ZINC_graph.json')
    lines.append(f"| {'Model':<15} | {'ZINC MAE':>15} |")
    lines.append("|" + "-"*17 + "|" + "-"*17 + "|")
    lines.append(f"| {'Paper GHC':<15} | {'0.337±0.020':>15} |")
    if r:
        lines.append(f"| {'Our GHC':<15} | {fmt(r['test_mean'], r['test_std'], is_mae=True):>15} |")
    
    # Check for improved ZINC
    r_imp = load_result('GHC_ZINC_improved.json')
    if r_imp:
        lines.append(f"| {'Our GHC (imp)':<15} | {fmt(r_imp['test_mean'], r_imp['test_std'], is_mae=True):>15} |")
    
    lines.append("")
    
    # ============================================================
    # Table 5: Ablation Study
    # ============================================================
    ablation_path = os.path.join(results_dir, 'ablation_table5.json')
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            abl = json.load(f)
        
        lines.append("# Table 5: Ablation Study (GHC)")
        lines.append("")
        lines.append(f"| {'Hyperparameter':<25} | {'Cora Δ':>10} | {'Paper Δ':>10} | {'RE Δ':>10} | {'Paper Δ':>10} |")
        lines.append("|" + "-"*27 + "|" + ("-"*12 + "|") * 4)
        
        cora_base = abl.get('cora_base', 'N/A')
        re_base = abl.get('re_base', 'N/A')
        lines.append(f"| {'Base':<25} | {cora_base:>10.2f} | {'78.85':>10} | {re_base:>10.2f} | {'92.27':>10} |")
        
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
            cora_key = f'cora_{name}'
            re_key = f're_{name}'
            cd = abl.get(cora_key, {}).get('delta', None)
            rd = abl.get(re_key, {}).get('delta', None)
            cd_str = f"{cd:+.2f}" if cd is not None else "N/A"
            rd_str = f"{rd:+.2f}" if rd is not None else "N/A"
            lines.append(f"| {name:<25} | {cd_str:>10} | {paper_cd:>+10.2f} | {rd_str:>10} | {paper_rd:>+10.2f} |")
        
        lines.append("")
    
    # ============================================================
    # Baselines
    # ============================================================
    lines.append("# Baselines")
    lines.append("")
    
    baselines = [
        ('GCN', 'Cora', '78.43±0.85'),
        ('GCN', 'CiteSeer', '66.75±1.86'),
        ('GCN', 'PubMed', '75.62±2.24'),
        ('MLP', 'Cora', '56.29±2.08'),
        ('MLP', 'Chameleon', '45.57±1.77'),
    ]
    
    lines.append(f"| {'Model':<10} | {'Dataset':<12} | {'Paper':>15} | {'Ours':>15} |")
    lines.append("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|")
    
    for model, dataset, paper_val in baselines:
        r = load_result(f'{model}_{dataset}_trans.json')
        if r:
            ours = fmt(r['test_mean'], r['test_std'])
        else:
            ours = "N/A"
        lines.append(f"| {model:<10} | {dataset:<12} | {paper_val:>15} | {ours:>15} |")
    
    lines.append("")
    
    # Write output
    output = '\n'.join(lines)
    
    with open(os.path.join(results_dir, 'results_tables.md'), 'w') as f:
        f.write(output)
    
    print(output)

if __name__ == '__main__':
    generate_all_tables()
