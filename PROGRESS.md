# HyperAggregation Replication Progress

## Current Phase
Implementing core model architecture (HyperAggregation, GHC, GHM, baselines, training pipeline).
datasets.py is done and tested.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): Load all datasets, handle splits - DONE, TESTED
- [ ] 2. HyperAggregation module (hyper_aggregation.py)
- [ ] 3. GraphHyperConv model (ghc.py)
- [ ] 4. GraphHyperMixer model (ghm.py)
- [ ] 5. Baseline models: GCN, MLP (baselines.py)
- [ ] 6. Training pipeline (train.py): Loss, optimizer, evaluation loop
- [ ] 7. Experiment runner (run_experiments.py): configs, seeds, splits
- [ ] 8. Run key experiments (focus on Cora, Roman-Empire, ZINC as representative)
- [ ] 9. Results collection, tables, REPORT.md
- [ ] 10. reproduce.sh

## Key Architecture Details (from paper)

### HyperAggregation (Section 3.1)
- Input: X_N(v) ∈ R^{|N(v)| × h}
- W_A ∈ R^{h×h}, W_B ∈ R^{h×m} (trainable)
- W_tar = GeLU(X · W_A) · W_B → R^{|N(v)| × m}
- HA(X) = (GeLU(X^T · W_tar) · W_tar^T)^T → R^{|N(v)| × h}
- Optional: input activation σ before HA
- Optional: LayerNorm + dropout before target network (trans_HA_input)
- Optional: LayerNorm + dropout after target network (trans_HA_output)

### GHC Block (Section 3.2)
- X^{i+1} = FF(HA(FF(X^i), A))
- Uses adjacency for neighborhood
- After HA: either root vertex embedding or mean-pool over neighborhood
- Optional root connection: concat root embedding with HA output
- Optional residual connections

### GHM Block (Section 3.2)
- Sample k-hop neighborhood, treat as fully connected
- X_N^{i+1} = FF(HA(FF(X_N^i)))
- Neighborhood size constant across blocks

### Hyperparameters (from paper + appendix)
- Cora GHC: m=64, hidden=256, dropout=0.6, self-loops=yes, residual=no, mean_agg=yes, root_conn=yes, trans_HA_output=yes
- Roman-Empire GHC: m=32, mix_dropout=0.1, hidden=256, dropout=0.3, self-loops=no, undirected=yes, residual=yes, mean_agg=no, root_conn=yes, trans_HA_input=yes
- General: Adam optimizer, lr typically 0.001-0.01, weight decay varies
- Depth: typically 2-4 blocks
- Seeds: 10 for small datasets, 3 for large

### Key Results to Reproduce (Tables 1-3)
- GHC Cora transductive: 78.85 ± 2.14
- GHC Roman-Empire transductive: 92.27 ± 0.57
- GHC ZINC: 0.337 ± 0.020
- GHC Photo transductive: 91.63 ± 0.79
- GHM Actor transductive: 37.56 ± 1.26

## Completed Work
- datasets.py: Loads Cora, CiteSeer, PubMed, Computers, Photo, Actor, Chameleon, Squirrel, Minesweeper, Roman-Empire, MNIST, CIFAR10, ZINC. Generates splits. Tested and working.

## Failed Approaches
(none yet)

## Rubric Status
- [x] Data Pipeline - datasets.py working
- [ ] Model Architecture - need hyper_aggregation.py, ghc.py, ghm.py, baselines.py
- [ ] Training Pipeline - need train.py
- [ ] Evaluation - need run_experiments.py
- [ ] Results Reproduction - need to run experiments and collect results
- [ ] reproduce.sh
- [ ] REPORT.md
