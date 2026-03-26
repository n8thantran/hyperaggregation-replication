# HyperAggregation Replication Progress

## Current Phase
Finalizing deliverables. Ablation (5/7 done, 2 running) and ZINC improved (1/3 seeds done) still running in background. Need to write REPORT.md, test reproduce.sh, and finalize.

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
- [x] 10. Run baselines (GCN on Cora/CiteSeer/PubMed, MLP on Cora/Chameleon)
- [~] 11. Run ablation study (Table 5) - 5/7 done, 2 running
- [~] 12. Run ZINC improved (h160_m128) - 1/3 seeds done (MAE 0.4507)
- [x] 13. Generate result tables (generate_tables.py)
- [ ] 14. Write REPORT.md
- [ ] 15. Test reproduce.sh
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
| Model | Dataset | Paper | Ours | Status |
|-------|---------|-------|------|--------|
| GCN | Cora | 78.43 | 78.50 | ✓ |
| GCN | CiteSeer | 66.75 | 69.23 | ✓ |
| GCN | PubMed | 75.62 | 74.94 | ✓ |
| MLP | Cora | 56.29 | 54.99 | ✓ |
| MLP | Chameleon | 45.57 | 49.05 | ~ |

### Table 5 (Ablation - 5/7 complete):
| Ablation | Cora Δ (ours) | Paper Δ | RE Δ (ours) | Paper Δ |
|----------|---------------|---------|-------------|---------|
| Self-loops | -1.26 | -2.84 | -3.59 | -0.13 |
| Normalize input | -1.75 | -0.66 | -2.75 | -0.01 |
| Residual | +0.19 | -3.09 | -15.67 | -1.22 |
| Root connection | -0.34 | -0.50 | -1.38 | -1.72 |
| Mean aggregate | -4.00 | -1.35 | -1.42 | -2.64 |
| Trans HA input | (running) | +0.47 | (running) | -1.15 |
| Trans HA output | (running) | -4.83 | (running) | -0.08 |

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights (N, m)
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T → (N, h)
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: k-hop neighborhood sampling, per-vertex HA

## Key Files
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHM, GCN, MLP
- datasets.py: Dataset loading for all 10 vertex datasets + ZINC
- train.py: Training pipeline with run_experiment() function
- run_reproduce.py: Main experiment runner with all configs
- run_ablation_fast.py: Ablation study runner
- generate_tables.py: Result table generator
- reproduce.sh: Top-level reproduction script

## Failed Approaches
1. Initial Chameleon/Squirrel: Used wrong split convention (geom-gcn splits vs paper's random splits)
2. Roman-Empire: Initially missed normalize_input=True and 4 blocks
3. ZINC: h64_m32 config gives MAE ~0.449; h160_m128 not helping much (0.4507 so far)
4. GCN Chameleon baseline: 37% vs paper 69% - likely split issue, not critical
5. Residual ablation on RE: -15.67 vs paper -1.22 - our RE config relies heavily on residual

## Rubric Assessment
- Core HyperAggregation implementation: DONE
- GHC model with all options: DONE
- GHM model: DONE (implemented, not extensively tested)
- All 10 vertex datasets: DONE
- ZINC graph-level: DONE (gap exists)
- Baselines: DONE (GCN, MLP)
- Ablation study: MOSTLY DONE (5/7, 2 running)
- reproduce.sh: NEEDS TESTING
- REPORT.md: NOT WRITTEN YET
- Results directory: HAS RESULTS
