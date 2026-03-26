# HyperAggregation Replication Progress

## Current Phase
Finalizing results and creating deliverables. Need to:
1. Run remaining experiments (CiteSeer, Computers, Photo with trans_input=True fix)
2. Run ZINC with correct hidden_dim=64 (paper uses 500K param budget)
3. Generate final tables, REPORT.md, reproduce.sh
4. Final git push

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits
- [x] 2. HyperAggregation module (in models.py)
- [x] 3. GHC model (in models.py)
- [x] 4. GHM model (in models.py)
- [x] 5. Baseline models GCN, MLP (in models.py)
- [x] 6. Training pipeline (train.py)
- [x] 7. Experiment runner (run_all.py)
- [x] 8. Run transductive vertex-level experiments - 17 results saved
- [x] 9. Run ZINC graph-level experiment - 3 seeds done (MAE ~0.449)
- [ ] 10. Re-run key experiments with corrected configs (trans_input=True)
- [ ] 11. Generate result tables, REPORT.md, reproduce.sh
- [ ] 12. Final git push

## Current Results vs Paper (Latest)

### Table 1 (Homophilic, transductive):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| Cora | 78.85±2.14 | 80.00±1.80 | ✓ Close (actually better) |
| CiteSeer | 66.82±1.66 | 67.16±1.33 | ✓ Close |
| PubMed | 76.31±2.71 | 73.97±1.59 | ~2.3% gap |
| Computers | 82.12±1.91 | 80.44±0.74 | ~1.7% gap |
| Photo | 91.63±0.79 | 91.33±0.64 | ✓ Close |

### Table 2 (Heterophilic, transductive):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| Chameleon | 74.78±1.82 | 68.86±1.42 | ~6% gap |
| Squirrel | 62.90±1.47 | 55.27±1.41 | ~7% gap |
| Actor | 36.40±1.46 | 37.00±1.03 | ✓ Close |
| Minesweeper | 87.49±0.61 | 86.17±0.80 | ✓ Close |
| Roman-Empire | 92.27±0.57 | 85.78±0.18 | ~6.5% gap |

### Table 3 (Graph-level):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| ZINC | 0.337±0.020 | 0.449±0.022 | ~0.11 gap |

### Baselines:
| Dataset | Paper GCN | My GCN | Paper MLP | My MLP |
|---------|-----------|--------|-----------|--------|
| Cora | 78.43 | 78.50 ✓ | 56.29 | 54.99 ✓ |
| CiteSeer | 66.75 | 69.23 | - | - |
| PubMed | 75.62 | 74.94 ✓ | - | - |
| Chameleon | 69.63 | 37.06 ⚠ | 45.57 | 49.05 |

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: Sample k-hop neighborhood, FF → HA → FF (fully connected)

### Paper hyperparameters (from Section 4.2 and ablation table):
**Cora**: hidden=256, mix=64, blocks=2, dropout=0.6, no residual, yes root_conn, yes mean_agg, self-loops=yes, trans_input=True, trans_output=True
**Roman-Empire**: hidden=256, mix=32, blocks=4, dropout=0.3, residual=yes, root_conn=yes, mean_agg=no, normalize_input=yes, undirected=yes, trans_input=yes, trans_output=no, self-loops=no, mix_dropout=0.1

## Key Config Fix Applied
- trans_input was False for Cora/Planetoid → changed to True (ablation table confirms base has trans_input=True)
- mix_dropout was 0.0 for Roman-Empire → changed to 0.1

## Files
- datasets.py: All dataset loading
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GCN, MLP
- train.py: Training pipeline with early stopping, LR scheduling
- run_final.py: Final experiment configs with corrected hyperparameters
- run_all.py, run_fast.py, run_reproduce.py: Various experiment runners
- results/: All JSON result files

## Known Gaps
1. Chameleon/Squirrel: ~6-7% gap. Paper uses WikipediaNetwork from torch_geometric which may have different splits. The paper's GCN gets 69.63% on Chameleon but my GCN gets 37.06% - this suggests a dataset/split issue.
2. Roman-Empire: ~6.5% gap. Could be due to missing implementation details.
3. ZINC: ~0.11 MAE gap. May need more training or different hidden_dim.
4. PubMed: ~2.3% gap. Moderate.

## Failed Approaches
- GCN on Chameleon gives 37% (paper: 69.63%) → suggests dataset split issue, not model issue
- Roman-Empire with mix_dropout=0.1 didn't help much (85.78 vs 85.72 without)
- ZINC with hidden_dim=256 gives ~0.449 MAE

## Rubric Status
- [x] Core method implemented (HyperAggregation, GHC)
- [x] Multiple datasets tested (10 datasets)
- [x] Baselines implemented (GCN, MLP)
- [x] Results close on 5+ datasets (Cora, CiteSeer, Photo, Actor, Minesweeper)
- [ ] reproduce.sh script
- [ ] REPORT.md
- [ ] Final cleanup and push
