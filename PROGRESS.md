# HyperAggregation Replication Progress

## Current Phase
Reading paper and creating implementation plan.

## Implementation Plan
- [ ] 1. Data pipeline (datasets.py): Load all datasets, handle splits
- [ ] 2. HyperAggregation module (hyper_aggregation.py)
- [ ] 3. GraphHyperConv model (ghc.py)
- [ ] 4. GraphHyperMixer model (ghm.py)
- [ ] 5. Baseline models: GCN, MLP (baselines.py)
- [ ] 6. Training pipeline (train.py): Loss, optimizer, evaluation loop
- [ ] 7. Experiment runner (run_experiments.py): configs, seeds, splits
- [ ] 8. Ablation studies
- [ ] 9. Results collection and reporting

## Key Decisions
### HyperAggregation
- W_A: h×h, W_B: h×m → W_tar: |N(v)| × m
- W_tar = GeLU(X·W_A)·W_B
- HA(X) = (GeLU(X^T·W_tar)·W_tar^T)^T
- Optional: input activation, dropout, layer norm before/after target

### GHC (GraphHyperConv)
- Block: FF → HA(adjacency) → FF
- After aggregation: root vertex or mean-pool
- Root connection: concat root embedding with HA output
- Residual connections configurable

### GHM (GraphHyperMixer)
- Sample k-hop neighborhood, treat as fully connected
- Block: FF → HA → FF
- Neighborhood size constant across blocks

### Hyperparameters from paper
- Cora: m=64, hidden=256, model_dropout=0.6, self-loops=yes, residual=no, mean_agg=yes, root_conn=yes, trans_HA_output=yes (norm/dropout after)
- Roman-Empire: m=32, mix_dropout=0.1, hidden=256, model_dropout=0.3, self-loops=no, undirected=yes, residual=yes, mean_agg=no, root_conn=yes, trans_HA_input=yes (norm/dropout before)

### Datasets
- Homophilic: Cora, CiteSeer, PubMed, OGB arXiv, Computers, Photo
- Heterophilic: Actor, Chameleon, Squirrel, Minesweeper, Roman-Empire
- Graph-level: MNIST, CIFAR10, ZINC (10k/1k/1k)

### Splits
- Cora/CiteSeer/PubMed/Computers/Photo: 10 random splits, 20 per class train, 30 val
- OGB arXiv: provided
- Actor: 48%/32%/20%
- Chameleon/Squirrel/Minesweeper/Roman-Empire: provided splits
- Seeds: 10 for smaller, 3 for larger (MNIST, CIFAR10, ZINC, arXiv)

### Inductive Setting
- Only training vertices/edges available
- For 20-per-class datasets: NOSMOG setup → 80%/20% unlabeled/test split of test data

## Completed Work
(none yet)

## Failed Approaches
(none yet)

## Rubric Status
- [ ] Data Pipeline
- [ ] Model Architecture
- [ ] Training Pipeline
- [ ] Evaluation
- [ ] Results Reproduction
