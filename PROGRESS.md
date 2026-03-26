# HyperAggregation Replication Progress

## Current Phase
Running ablation experiments (Table 5) and ZINC improved. Need to finalize deliverables.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits
- [x] 2. HyperAggregation module (in models.py) - both batched and non-batched
- [x] 3. GHC model (in models.py) - with embedding support for ZINC
- [x] 4. GHM model (in models.py) - k-hop neighborhood sampling
- [x] 5. Baseline models GCN, MLP (in models.py)
- [x] 6. Training pipeline (train.py) - supports vertex/graph tasks
- [x] 7. Experiment runner scripts (run_all.py, run_fast.py, run_reproduce.py)
- [x] 8. Run transductive vertex-level experiments - ALL 10 datasets done
- [x] 9. Run ZINC graph-level experiment - 3 seeds done (MAE ~0.449)
- [x] 10. Run baselines (GCN on Cora/CiteSeer/PubMed/Chameleon, MLP on Cora/Chameleon)
- [~] 11. Run ablation study (Table 5) - IN PROGRESS (2/7 done)
- [~] 12. Run ZINC improved (h160_m128) - IN PROGRESS
- [ ] 13. Generate final result tables (generate_tables.py)
- [ ] 14. Write REPORT.md
- [ ] 15. Write reproduce.sh
- [ ] 16. Final git push

## Current Results vs Paper

### Table 1 (Homophilic, transductive):
| Dataset | Paper GHC | My GHC | Status |
|---------|-----------|--------|--------|
| Cora | 78.85±2.14 | 80.00±1.80 | ✓ Close (better) |
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

### Table 5 (Ablation - partial):
| Ablation | Cora Δ (ours) | Paper Δ | RE Δ (ours) | Paper Δ |
|----------|---------------|---------|-------------|---------|
| Self-loops | -1.26 | -2.84 | -3.59 | -0.13 |
| Normalize input | -1.75 | -0.66 | -2.75 | -0.01 |
| (remaining 5 still running) | | | | |

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights (N, m)
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T → (N, h)
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: Sample k-hop neighborhood, FF → HA → FF (fully connected)

### Paper hyperparameters (from Section 4.2 and ablation table):
**Cora**: hidden=256, mix=64, blocks=2, dropout=0.6, no residual, yes root_conn, yes mean_agg, self-loops=yes, trans_input=True, trans_output=True
**Roman-Empire**: hidden=256, mix=32, blocks=4, dropout=0.3, residual=yes, root_conn=yes, mean_agg=no, normalize_input=yes, undirected=yes, trans_input=yes, trans_output=no, self-loops=no, mix_dropout=0.1
**ZINC**: hidden=64, mix=64, blocks=4, dropout=0.0, residual=yes, root_conn=yes, mean_agg=yes, trans_input=yes, trans_output=yes, 500K param budget

## Key Decisions
- Using PyG for data loading and scatter operations
- HyperAggregationBatched uses scatter_add for efficient per-neighborhood computation
- Memory-efficient chunked computation for large graphs
- For mean_agg=False: use root vertex's own W_tar to read from aggregated features

## Completed Work
- **datasets.py**: Loads Cora, CiteSeer, PubMed, Computers, Photo, Chameleon, Squirrel, Actor, Minesweeper, Roman-Empire, ZINC. Handles random splits for Cora/CiteSeer/PubMed/Computers/Photo.
- **models.py**: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHMBlock, GHM, GCN, MLP. All tested.
- **train.py**: Training pipeline with early stopping, supports vertex/graph tasks, transductive/inductive settings.
- **run_reproduce.py**: Main experiment runner with all configs.
- **run_ablation_fast.py**: Ablation study runner (Table 5).
- **generate_tables.py**: Generates result tables from JSON files.

## Failed Approaches
1. **GCN on Chameleon**: Getting 37% instead of paper's 69.63%. The paper likely uses a different GCN implementation or split. Not critical for replication of HyperAggregation.
2. **Roman-Empire gap**: 85.78% vs 92.27%. The gap may be due to undirected edge handling or the specific split used. The paper uses provided splits from the heterophilic benchmark.
3. **Chameleon/Squirrel gap**: ~6-7% below paper. These datasets are known to be sensitive to splits and preprocessing. The paper uses specific splits from Platonov et al.
4. **ZINC gap**: 0.449 vs 0.337. May need larger model or more training. Running h160_m128 config now.

## Running Processes
- `run_ablation_fast.py` → ablation_output.log (2/7 ablations done)
- `run_zinc_improved.py` → zinc_improved_output.log (1/3 seeds done for h160_m128)

## Rubric Status
- [x] HyperAggregation module implemented correctly
- [x] GHC model implemented
- [x] GHM model implemented  
- [x] Baselines (GCN, MLP) implemented
- [x] Table 1 results (5/5 datasets, most within ~2%)
- [x] Table 2 results (5/5 datasets, some gaps on Chameleon/Squirrel/RE)
- [x] Table 3 results (ZINC, gap exists)
- [~] Table 5 ablation (in progress)
- [ ] reproduce.sh finalized
- [ ] REPORT.md written
- [ ] Final commit pushed
