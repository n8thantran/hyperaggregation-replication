# HyperAggregation Replication Progress

## Current Phase
Running experiments across multiple datasets. Need to generate comprehensive results.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits - DONE, TESTED
- [x] 2. HyperAggregation module (in models.py) - DONE, VERIFIED correct against non-batched
- [x] 3. GHC model (in models.py) - DONE
- [x] 4. GHM model (in models.py) - DONE
- [x] 5. Baseline models GCN, MLP (in models.py) - DONE
- [x] 6. Training pipeline (train.py) - DONE
- [x] 7. Experiment runner (run_experiments.py) - DONE
- [ ] 8. Run key experiments across datasets
- [ ] 9. Results collection, tables, REPORT.md
- [ ] 10. reproduce.sh

## Key Results So Far
- GHC Cora transductive: 78.01 ± 1.88 (paper: 78.85 ± 2.14) ✓ CLOSE
- GHC Roman-Empire transductive: ~86% single seed (paper: 92.27 ± 0.57) - GAP REMAINS
  - Saved result file is outdated (76.42%) from before fixing mean_agg

## Roman-Empire Investigation
- Verified HA implementation is mathematically correct (batched matches non-batched)
- Best config: hidden=256, mix_dim=32, blocks=4-6, dropout=0.3, mix_dropout=0.1, 
  trans_input=True, mean_agg=False, root_conn=True, residual=True, no self-loops, 
  make_undirected=True, normalize_input=True
- Tried: different depths (2-6), weight_decay (0, 1e-4, 5e-4), 1000 epochs → all plateau ~86%
- May be missing: block-level LayerNorm, specific LR schedule, or other architectural detail
- DECISION: Accept ~86% gap and move on to other experiments for broader coverage

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: Sample k-hop neighborhood, FF → HA → FF (fully connected)

### Key hyperparameters per dataset:
- Cora: m=64, hidden=256, dropout=0.6, self-loops=yes, residual=no, mean_agg=yes, root_conn=yes, trans_output=yes
- Roman-Empire: m=32, hidden=256, dropout=0.3, mix_dropout=0.1, self-loops=no, undirected=yes, residual=yes, mean_agg=no, root_conn=yes, trans_input=yes

## Files
- datasets.py: All dataset loading
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHMBlock, GHM, GCN, MLP
- train.py: Training/eval loops, run_single_experiment, run_experiment
- run_experiments.py: Experiment configs

## Failed Approaches
- Roman-Empire: Tried depths 2-16, various weight decays, input_activation, different epochs → plateau at ~86%
- The gap to 92.27% likely requires some undisclosed architectural detail (block-level normalization pattern)
