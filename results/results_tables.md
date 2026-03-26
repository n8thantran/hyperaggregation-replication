# Replication Results

## Table 1: Homophilic Datasets (Transductive, Accuracy %)

| Dataset | Paper GHC | My GHC |
|---------|-----------|--------|
| Cora | 78.85±2.14 | 78.01±1.88 |
| CiteSeer | 66.82±1.66 | 67.16±1.33 |
| PubMed | 76.31±2.71 | 73.34±1.61 |
| Computers | 82.12±1.91 | 80.44±0.74 |
| Photo | 91.63±0.79 | 91.33±0.64 |

## Table 2: Heterophilic Datasets (Transductive, Accuracy %)

| Dataset | Paper GHC | My GHC |
|---------|-----------|--------|
| Chameleon | 74.78±1.82 | 68.86±1.42 |
| Squirrel | 62.90±1.47 | 55.27±1.41 |
| Actor | 36.40±1.46 | 37.00±1.03 |
| Minesweeper | 87.49±0.61 | 86.17±0.80 |
| RomanEmpire | 92.27±0.57 | 81.15±0.49 |

## Table 3: Graph-Level (ZINC MAE, lower is better)

| ZINC | 0.337±0.020 | Not yet available |

## Baselines

| Model/Dataset | Paper | Mine |
|--------------|-------|------|
| GCN Cora | 78.43±0.85 | 78.50±0.23 |
| GCN CiteSeer | 66.75±1.86 | 69.23±0.51 |
| GCN PubMed | 75.62±2.24 | 74.94±0.24 |
| GCN Chameleon | 69.63±2.41 | 37.06±3.43 |
| MLP Cora | 56.29±2.08 | 54.99±0.32 |
| MLP Chameleon | 45.57±1.77 | 49.05±1.44 |
