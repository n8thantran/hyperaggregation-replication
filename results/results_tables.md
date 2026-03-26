# Replication Results

## Table 1: Homophilic Datasets (Transductive)

| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |
|---------|-----------|---------|-----------|----------|
| Cora | 78.85 ± 2.14 | 78.01 ± 1.88 | 78.43 ± 1.35 | 78.50 ± 0.23 |
| CiteSeer | 66.82 ± 1.66 | 67.16 ± 1.33 | 66.75 ± 1.89 | 69.23 ± 0.51 |
| PubMed | 76.31 ± 2.71 | 73.34 ± 1.61 | 75.62 ± 2.28 | 74.94 ± 0.24 |
| Computers | 82.12 ± 1.91 | 80.44 ± 0.74 | 80.72 ± 1.79 | N/A |
| Photo | 91.63 ± 0.79 | 91.33 ± 0.64 | 91.16 ± 0.83 | N/A |

## Table 2: Heterophilic Datasets (Transductive)

| Dataset | Paper GHC | Our GHC |
|---------|-----------|----------|
| Chameleon | 74.78 ± 1.82 | 68.86 ± 1.42 |
| Squirrel | 62.90 ± 1.47 | 55.27 ± 1.41 |
| Actor | 36.40 ± 1.46 | 37.00 ± 1.03 |
| Minesweeper | 87.49 ± 0.61 | 86.17 ± 0.80 |
| RomanEmpire | 92.27 ± 0.57 | 81.15 ± 0.49 |

## Table 3: Graph-Level (ZINC)

| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |
|---------|------------------|------------------|
| ZINC | 0.337 ± 0.020 | 0.584 ± 0.000 |

## All Results

- **GCN_Chameleon_trans**: 0.3706 ± 0.0343
- **GCN_CiteSeer_trans**: 0.6923 ± 0.0051
- **GCN_Cora_trans**: 0.7850 ± 0.0023
- **GCN_PubMed_trans**: 0.7494 ± 0.0024
- **GHC_Actor_trans**: 0.3700 ± 0.0103
- **GHC_Chameleon_trans**: 0.6886 ± 0.0142
- **GHC_CiteSeer_trans**: 0.6716 ± 0.0133
- **GHC_Computers_trans**: 0.8044 ± 0.0074
- **GHC_Cora_trans**: 0.7801 ± 0.0188
- **GHC_Minesweeper_trans**: 0.8617 ± 0.0080
- **GHC_Photo_trans**: 0.9133 ± 0.0064
- **GHC_PubMed_trans**: 0.7334 ± 0.0161
- **GHC_RomanEmpire_trans**: 0.8115 ± 0.0049
- **GHC_Squirrel_trans**: 0.5527 ± 0.0141
- **GHC_ZINC_graph**: 0.5838 ± 0.0000
- **MLP_Chameleon_trans**: 0.4905 ± 0.0144
- **MLP_Cora_trans**: 0.5499 ± 0.0032
