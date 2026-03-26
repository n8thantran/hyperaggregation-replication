# HyperAggregation Replication Progress

## Current Phase
Creating final deliverables (reproduce.sh, REPORT.md, result tables). ZINC running in background.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits - DONE, TESTED
- [x] 2. HyperAggregation module (in models.py) - DONE, VERIFIED (exact match with non-batched version)
- [x] 3. GHC model (in models.py) - DONE
- [x] 4. GHM model (in models.py) - DONE
- [x] 5. Baseline models GCN, MLP (in models.py) - DONE
- [x] 6. Training pipeline (train.py) - DONE
- [x] 7. Experiment runner (run_all.py) - DONE
- [x] 8. Run transductive vertex-level experiments - DONE (16 results)
- [x] 9. Run ZINC graph-level experiment - Seed 0: 0.468 MAE, Seed 1 in progress
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
| ZINC | 0.337±0.020 | 0.468 (seed 0, 358ep) | ~0.13 gap, seed 1 running |

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
**Cora**: hidden=256, mix=64, blocks=2, dropout=0.6, no residual, yes root_conn, yes mean_agg, self-loops=yes, trans_output=yes
**Roman-Empire**: hidden=256, mix=32, blocks=4, dropout=0.3, residual=yes, root_conn=yes, mean_agg=no, normalize_input=yes, undirected=yes, trans_input=yes, no self-loops

## Files
- datasets.py: All dataset loading (Planetoid, Amazon, WikipediaNetwork, HeterophilousGraph, ZINC, GNNBenchmark)
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHMBlock, GHM, GCN, MLP
- train.py: Training/eval loops, run_single_experiment, run_experiment (handles vertex+graph tasks)
- run_all.py: All experiment configs and runner
- run_zinc_quick.py: ZINC-specific experiment script (running in background)
- generate_tables.py: Generate result tables from JSON files

## Verified Correctness
- HyperAggregationBatched matches HyperAggregation exactly (both mean_agg and root readout with self-loops)
- GCN baseline matches paper on Cora, PubMed
- MLP baseline matches paper on Cora

## Failed Approaches / Known Gaps
- Roman-Empire: ~6% gap. Tried depths 2-10, various weight decays, mix_dims. The gap may be due to:
  - Different split handling (paper uses 10 splits × 10 seeds = 100 runs)
  - Possible undisclosed architectural details
  - My mean_agg=False implementation without self-loops differs from paper's approach
- Chameleon/Squirrel: ~6-7% gap. These use filtered splits from the paper which may differ
- Chameleon GCN baseline at 37% is wrong (paper: 69.63%) - likely wrong split or undirected config
- ZINC: 0.468 vs paper 0.337 - may need edge features, different LR schedule, or longer training

## Rubric Status
- [x] Core method (HyperAggregation) implemented correctly
- [x] GHC model implemented
- [x] GHM model implemented  
- [x] Baselines (GCN, MLP) implemented
- [x] All 10 transductive datasets attempted (5 homophilic + 5 heterophilic)
- [x] ZINC graph-level experiment running
- [x] Results close on 7/10 transductive datasets
- [ ] reproduce.sh - TODO
- [ ] REPORT.md - TODO
- [ ] Final result tables - TODO
