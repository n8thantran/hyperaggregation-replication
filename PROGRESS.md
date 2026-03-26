# HyperAggregation Replication Progress

## Current Phase
Running experiments across all datasets to reproduce Tables 1, 2, 3 from paper.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits - DONE, TESTED
- [x] 2. HyperAggregation module (in models.py) - DONE, VERIFIED
- [x] 3. GHC model (in models.py) - DONE
- [x] 4. GHM model (in models.py) - DONE
- [x] 5. Baseline models GCN, MLP (in models.py) - DONE
- [x] 6. Training pipeline (train.py) - DONE
- [x] 7. Experiment runner (run_all.py) - DONE
- [ ] 8. Run all key experiments
- [ ] 9. Results collection, tables, REPORT.md
- [ ] 10. reproduce.sh

## Key Results So Far
- GHC Cora transductive: 78.01 ± 1.88 (paper: 78.85 ± 2.14) ✓ CLOSE
- GHC Roman-Empire transductive: 85.74 ± 0.17 (paper: 92.27 ± 0.57) - GAP (accepted, move on)

## Roman-Empire Investigation
- Tried: depths 2-10, weight_decay (0, 1e-4, 5e-4, 1e-3), various dropout, mix_dim, hidden_dim
- Best: blocks=10 gives ~86.7%, blocks=4 gives ~85.7%
- DECISION: Accept ~86% gap. The paper's 92.27% likely requires undisclosed architectural details.
- Moving on to maximize dataset coverage.

## Key Architecture Details (from paper)
### HyperAggregation (Section 3.1)
- W_tar = GeLU(X · W_A) · W_B → target network weights
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T
- Optional: LayerNorm+dropout before/after target network

### GHC Block: FF → HA(A) → FF with optional root_conn, residual
### GHM Block: Sample k-hop neighborhood, FF → HA → FF (fully connected)

### Key paper results to reproduce:
**Table 1 (Homophilic, transductive):**
- Cora: GHC=78.85, GCN=78.43, MLP=56.29
- CiteSeer: GHC=66.82
- PubMed: GHC=76.31
- Computers: GHC=82.12
- Photo: GHC=91.63
**Table 2 (Heterophilic, transductive):**  
- Chameleon: GHC=74.78, Squirrel: GHC=62.90, Actor: GHC=36.40
- Minesweeper: GHC=87.49, Roman-Empire: GHC=92.27
**Table 3 (Graph-level):**
- ZINC: GHC=0.337

## Files
- datasets.py: All dataset loading
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GHMBlock, GHM, GCN, MLP
- train.py: Training/eval loops, run_single_experiment, run_experiment
- run_all.py: All experiment configs and runner

## Failed Approaches
- Roman-Empire: Tried depths 2-10, various weight decays, mix_dims, hidden_dims → plateau at ~86%
- The gap to 92.27% likely requires some undisclosed architectural detail
