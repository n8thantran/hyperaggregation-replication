# HyperAggregation Replication Progress

## Current Phase
Finishing ZINC experiment and generating deliverables (reproduce.sh, REPORT.md).

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits - DONE, TESTED
- [x] 2. HyperAggregation module (in models.py) - DONE, VERIFIED
- [x] 3. GHC model (in models.py) - DONE
- [x] 4. GHM model (in models.py) - DONE
- [x] 5. Baseline models GCN, MLP (in models.py) - DONE
- [x] 6. Training pipeline (train.py) - DONE
- [x] 7. Experiment runner (run_all.py) - DONE
- [x] 8. Run transductive vertex-level experiments - DONE (16 results)
- [ ] 9. Run ZINC graph-level experiment - IN PROGRESS (1 seed done: 0.535 MAE at 100 epochs, need more training)
- [ ] 10. Generate result tables, REPORT.md, reproduce.sh
- [ ] 11. Final git push

## Current Results vs Paper

### Table 1 (Homophilic, transductive):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| Cora | 78.85±2.14 | 78.01±1.88 | ✓ Close |
| CiteSeer | 66.82±1.66 | 67.16±1.33 | ✓ Close |
| PubMed | 76.31±2.71 | 73.34±1.61 | ~3% gap |
| Computers | 82.12±1.91 | 80.44±0.74 | ~2% gap |
| Photo | 91.63±0.79 | 91.33±0.64 | ✓ Close |

### Table 2 (Heterophilic, transductive):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| Chameleon | 74.78±1.82 | 68.86±1.42 | ~6% gap |
| Squirrel | 62.90±1.47 | 55.27±1.41 | ~7% gap |
| Actor | 36.40±1.46 | 37.00±1.03 | ✓ Close |
| Minesweeper | 87.49±0.61 | 86.17±0.80 | ✓ Close |
| Roman-Empire | 92.27±0.57 | 85.72±0.10 | ~6% gap |

### Table 3 (Graph-level):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| ZINC | 0.337±0.020 | 0.535 (1 seed, 100ep) | IN PROGRESS - needs more training |

### Baselines:
| Dataset | Paper GCN | My GCN | Paper MLP | My MLP |
|---------|-----------|--------|-----------|--------|
| Cora | 78.43 | 78.50 ✓ | 56.29 | 54.99 ✓ |
| CiteSeer | 66.75 | 69.23 | - | - |
| PubMed | 75.62 | 74.94 ✓ | - | - |
| Chameleon | 69.63 | 37.06 ⚠ WRONG | 45.57 | 49.05 |

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: Sample k-hop neighborhood, FF → HA → FF (fully connected)

### Paper hyperparameters (from Section 4.2 and ablation table):
**Cora**: hidden=256, mix=64, blocks=2, dropout=0.6, no residual, yes root_conn, yes mean_agg, self-loops=yes, trans_output=yes
**Roman-Empire**: hidden=256, mix=32, blocks=4, dropout=0.3, residual=yes, root_conn=yes, mean_agg=no, normalize_input=yes, undirected=yes, trans_input=yes, no self-loops

## Files
- datasets.py: All dataset loading (Planetoid, Amazon, WikipediaNetwork, HeterophilousGraph, ZINC, GNNBenchmark)
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHMBlock, GHM, GCN, MLP
- train.py: Training/eval loops, run_single_experiment, run_experiment (handles vertex+graph tasks)
- run_all.py: All experiment configs and runner
- run_zinc.py: ZINC-specific experiment script

## Failed Approaches
- Roman-Empire: Tried depths 2-10, various weight decays, mix_dims, hidden_dims → plateau at ~86%
- The gap to 92.27% likely requires some undisclosed architectural detail or hyperparameter
- Chameleon GCN at 37% is wrong - likely using wrong splits or undirected config
- ZINC: 100 epochs gives only 0.535 MAE. Need 300-500+ epochs. ~3s/epoch = 15-25 min per seed.

## Priority for Remaining Time
1. Run ZINC for longer (300 ep, 3 seeds) - launch in background
2. Generate deliverables (reproduce.sh, REPORT.md, all results tables)
3. Final git push and end task
