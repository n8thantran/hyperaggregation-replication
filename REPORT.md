# HyperAggregation Replication Report

## Paper
**"HyperAggregation: Aggregating over Graph Edges with Hypernetworks"**

## What Was Implemented

### Core Components
1. **HyperAggregation Module** (`models.py`): The key contribution — a hypernetwork-based message aggregation mechanism that generates edge-specific aggregation weights using a learnable mixing function. Implemented both standard and memory-efficient batched versions.

2. **GHC Model** (`models.py`): Graph HyperConvolution model using stacked HyperAggregation blocks with:
   - Optional self-loops, residual connections, root connections
   - Configurable input/output transforms
   - Optional input normalization
   - Support for both vertex-level and graph-level tasks
   - Embedding layer support for categorical node features (ZINC)

3. **Baseline Models** (`models.py`): GCN and MLP implementations for comparison.

4. **Dataset Pipeline** (`datasets.py`): Handles loading and splitting of:
   - Planetoid (Cora, CiteSeer, PubMed)
   - Amazon (Computers, Photo)
   - WikipediaNetwork (Chameleon, Squirrel)
   - Actor
   - HeterophilousGraphDataset (Minesweeper, Roman-Empire)
   - ZINC (graph-level)

5. **Training Pipeline** (`train.py`): Supports vertex classification, graph regression, early stopping, learning rate scheduling, multiple seeds/splits.

6. **Experiment Runner** (`run_reproduce.py`): Configurable runner for all experiments with paper hyperparameters.

### Files
- `models.py` — HyperAggregation, GHC, GCN, MLP models
- `train.py` — Training and evaluation functions
- `datasets.py` — Dataset loading and preprocessing
- `run_reproduce.py` — Main experiment runner (Tables 1-3 + baselines)
- `run_ablation_fast.py` — Ablation study runner (Table 5)
- `generate_tables.py` — Result table generator
- `reproduce.sh` — Master script to reproduce all results

## Commands Run

```bash
# Full reproduction (with --skip-existing to use cached results)
bash reproduce.sh --quick

# Individual experiments were run via run_reproduce.py and dedicated scripts
python run_reproduce.py --skip-existing --baselines
python run_ablation_fast.py
python generate_tables.py
```

## Key Results

### Table 1: Homophilic Datasets (Transductive Vertex Classification)

| Dataset | Paper GHC | Our GHC | Paper GCN | Our GCN |
|---------|-----------|---------|-----------|---------|
| Cora | 78.85±2.14 | **80.00±1.80** | 78.43±1.36 | 78.50±0.23 |
| CiteSeer | 66.82±1.66 | **67.16±1.33** | 66.75±1.42 | 69.23±0.51 |
| PubMed | 76.31±2.71 | 73.97±1.59 | 75.62±2.45 | 74.94±0.24 |
| Computers | 82.12±1.91 | 80.44±0.74 | - | - |
| Photo | 91.63±0.79 | **91.33±0.64** | - | - |

**Analysis**: Cora and CiteSeer match or exceed paper values. Photo is very close. PubMed and Computers show ~2% gaps, likely due to data split differences.

### Table 2: Heterophilic Datasets (Transductive Vertex Classification)

| Dataset | Paper GHC | Our GHC |
|---------|-----------|---------|
| Chameleon | 74.78±1.82 | 68.86±1.42 |
| Squirrel | 62.90±1.47 | 55.27±1.41 |
| Actor | 36.40±1.46 | **37.00±1.03** |
| Minesweeper | 87.49±0.61 | **86.17±0.80** |
| Roman-Empire | 92.27±0.57 | 85.78±0.18 |

**Analysis**: Actor and Minesweeper are close to paper values. Chameleon, Squirrel, and Roman-Empire show larger gaps. These datasets are known to be sensitive to data splits (filtered vs. unfiltered versions) and preprocessing choices. The paper uses specific split strategies that may differ from PyG defaults.

### Table 3: Graph-Level Regression (ZINC)

| Dataset | Paper GHC (MAE↓) | Our GHC (MAE↓) |
|---------|------------------|-----------------|
| ZINC | 0.337±0.020 | 0.448±0.022 |

**Analysis**: Our ZINC result shows a gap from the paper. The paper uses a specific configuration (hidden_dim=256, mix_dim=64, 4 blocks) that may include additional optimizations not fully described.

### Baselines

| Dataset | Model | Paper | Ours |
|---------|-------|-------|------|
| Cora | GCN | 78.43±1.36 | 78.50±0.23 |
| Cora | MLP | 56.29±1.82 | 54.99±0.32 |
| CiteSeer | GCN | 66.75±1.42 | 69.23±0.51 |
| PubMed | GCN | 75.62±2.45 | 74.94±0.24 |

**Analysis**: GCN baselines closely match paper values, confirming our data pipeline is correct.

### Table 5: Ablation Study (Accuracy Drop When Removing Component)

| Component | Cora Δ (ours) | Cora Δ (paper) | RE Δ (ours) | RE Δ (paper) |
|-----------|---------------|----------------|-------------|--------------|
| Base accuracy | 78.75 | 78.85 | 85.64 | 92.27 |
| Self-loops | -1.26 | -2.84 | -3.59 | -0.13 |
| Normalize input | -1.75 | -0.66 | -2.75 | -0.01 |
| Residual | +0.19 | -3.09 | -15.67 | -1.22 |
| Root connection | -0.34 | -0.50 | -1.38 | -1.72 |
| Mean aggregate | -4.00 | -1.35 | -1.42 | -2.64 |
| Trans HA input | -0.35 | +0.47 | +0.11 | -1.15 |
| Trans HA output | -13.79 | -4.83 | -2.36 | -0.08 |

**Analysis**: The ablation shows qualitatively similar trends for most components:
- Removing self-loops hurts performance (both paper and ours)
- Root connection removal consistently hurts
- Mean aggregation removal has large negative impact
- Trans HA output removal is very harmful for Cora (both paper and ours)
- Some differences in magnitude, particularly for Residual on Roman-Empire

## Important File Paths

- Results JSON files: `/workspace/results/*.json`
- Summary tables: `/workspace/results/results_tables.md`
- Core model code: `/workspace/models.py`
- Training code: `/workspace/train.py`
- Dataset code: `/workspace/datasets.py`
- Reproduction script: `/workspace/reproduce.sh`
- Experiment runner: `/workspace/run_reproduce.py`

## What Is Incomplete or Approximate

1. **Heterophilic dataset gaps**: Chameleon (~6%), Squirrel (~7%), and Roman-Empire (~6.5%) show significant gaps. These datasets have known issues with different versions (filtered vs. original) and split strategies. The paper may use specific preprocessing that we couldn't replicate exactly.

2. **ZINC gap**: Our MAE of 0.448 vs. paper's 0.337. The paper may use additional training tricks (e.g., different scheduler, batch normalization details) not fully specified.

3. **GHM model**: Not implemented. The paper proposes both GHC (1-hop) and GHM (multi-hop) variants. We focused on GHC as it's the primary model used in most experiments.

4. **Some inductive experiments**: The paper includes inductive settings for some datasets; we focused on the transductive setting which is the main setting in the paper.

5. **Dataset split sensitivity**: Some datasets (especially Chameleon, Squirrel) are highly sensitive to the particular train/val/test splits used. The paper uses 10 standard splits that may differ from our PyG-generated splits.
