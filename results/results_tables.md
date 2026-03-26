# HyperAggregation Replication Results

Replication of 'HyperAggregation: Aggregating over Graph Edges with Hypernetworks'

## Table 1: Homophilic Datasets (Transductive Vertex Classification)

| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |
|---------|-----------|---------|-----------|---------|
| Cora | 78.85±2.14 | 80.00±1.80 | 78.43±1.36 | 78.50±0.23 |
| CiteSeer | 66.82±1.66 | 67.16±1.33 | 66.75±1.42 | 69.23±0.51 |
| PubMed | 76.31±2.71 | 73.97±1.59 | 75.62±2.45 | 74.94±0.24 |
| Computers | 82.12±1.91 | 80.44±0.74 | - | - |
| Photo | 91.63±0.79 | 91.33±0.64 | - | - |

## Table 2: Heterophilic Datasets (Transductive Vertex Classification)

| Dataset | Paper GHC | Our GHC |
|---------|-----------|---------|
| Chameleon | 74.78±1.82 | 68.86±1.42 |
| Squirrel | 62.90±1.47 | 55.27±1.41 |
| Actor | 36.40±1.46 | 37.00±1.03 |
| Minesweeper | 87.49±0.61 | 86.17±0.80 |
| Roman-Empire | 92.27±0.57 | 85.78±0.18 |

## Table 3: Graph-Level Regression (ZINC)

| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |
|---------|------------------|-----------------|
| ZINC | 0.337±0.020 | 0.448±0.022 |

## Baselines

| Dataset | Model | Paper | Ours |
|---------|-------|-------|------|
| Cora | GCN | 78.43±1.36 | 78.50±0.23 |
| Cora | MLP | 56.29±1.82 | 54.99±0.32 |
| CiteSeer | GCN | 66.75±1.42 | 69.23±0.51 |
| PubMed | GCN | 75.62±2.45 | 74.94±0.24 |
| Chameleon | GCN | 69.63±1.73 | 37.06±3.43 |
| Chameleon | MLP | 45.57±2.07 | 49.05±1.44 |