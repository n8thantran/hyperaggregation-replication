# HyperAggregation Replication Progress

## Current Phase
FINALIZING - All experiments complete, investigating accuracy gaps but core implementation done.

## Implementation Plan
- [x] 1. Data pipeline (datasets.py): All 10 vertex datasets + ZINC
- [x] 2. HyperAggregation module (models.py) - both batched and non-batched
- [x] 3. GHC model (models.py) - with embedding support for ZINC
- [x] 4. Baseline models GCN, MLP (models.py)
- [x] 5. Training pipeline (train.py) - vertex/graph tasks, early stopping
- [x] 6. Experiment runner scripts (run_reproduce.py, run_ablation_fast.py)
- [x] 7. All 10 transductive vertex-level experiments
- [x] 8. ZINC graph-level experiment (3 seeds, MAE ~0.448)
- [x] 9. Baselines (GCN, MLP on Cora/CiteSeer/PubMed/Chameleon)
- [x] 10. Ablation study (Table 5) - 7/7 ablations complete
- [x] 11. Result tables (generate_tables.py → results/results_tables.md)
- [x] 12. REPORT.md with analysis
- [x] 13. reproduce.sh verified working

## Results Summary
- Table 1 (Homophilic): 2/5 match or beat paper (Cora 80.0 vs 78.85, CiteSeer 72.7 vs 72.22)
  - PubMed close (73.97 vs 77.56), Photo close (92.67 vs 93.40), Computers gap (85.50 vs 88.17)
- Table 2 (Heterophilic): Actor 35.49 vs 35.49, Minesweeper 88.52 vs 89.93
  - Roman-Empire 85.78 vs 92.27 (~6.5% gap), Chameleon/Squirrel larger gaps
- Table 3 (ZINC): MAE 0.448 vs paper 0.337
- Baselines: GCN/MLP match paper well
- Table 5 (Ablation): All 7 ablations complete, qualitatively similar trends

## Key Findings on Gaps
- Roman-Empire gap (~6.5%): Investigated hyperparameter variations (blocks, mix_dim, input_act, lr, dropout) - none close the gap significantly. The model consistently gets ~86% vs paper's 92.27%. Likely due to subtle implementation differences or unreported hyperparameter details.
- Chameleon/Squirrel gaps: These are heterophilic datasets where the paper's provided splits differ. The model architecture is correct but may need exact hyperparameter tuning.
- ZINC gap: Graph-level task, MAE 0.448 vs 0.337. May need more epochs or learning rate schedule tuning.

## Key Files
- models.py: HyperAggregation, HyperAggregationBatched, GHCBlock, GHC, GCN, MLP
- train.py: Training pipeline with early stopping, vertex/graph tasks
- datasets.py: All dataset loading with proper splits
- run_reproduce.py: Main experiment runner with all configs
- run_ablation_fast.py: Ablation study runner
- generate_tables.py: Result table generation
- reproduce.sh: Master reproduction script
- results/: All JSON results and tables
- REPORT.md: Final report

## Architecture Details (from paper)
- Block: FF → HA → [root_conn] → FF
- HA: W_tar = σ(X·W_A)·W_B, output = GeLU(X^T·W_tar)·W_tar^T
- Root connection: concatenate HA output with input, apply GeLU
- Options: trans_input (LayerNorm+Dropout before HA), trans_output (LayerNorm+Dropout after HA)
- Options: mean_agg vs root readout, self-loops, normalize_input, residual

## Dataset Split Details (from paper)
- Cora/CiteSeer/PubMed/Computers/Photo: 10 random splits, 20 per class train, 30 val
- Actor: 48%/32%/20% random splits
- Chameleon/Squirrel/Minesweeper/Roman-Empire: provided 10 splits from heterophilic benchmark
- ZINC: standard 10k/1k/1k subset

## Failed Approaches
- Tried input_activation for Roman-Empire: no improvement
- Tried 6 blocks for Roman-Empire: marginal improvement
- Tried various mix_dim values: no significant change
- ZINC: tried different hidden/mix dims, learning rates - best was ~0.448
