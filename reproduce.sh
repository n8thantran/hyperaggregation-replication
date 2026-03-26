#!/bin/bash
# Reproduce key results from "HyperAggregation: Aggregating over Graph Edges with Hypernetworks"
#
# This script runs the main experiments from the paper:
# - Table 1: Homophilic vertex classification (Cora, CiteSeer, PubMed, Computers, Photo)
# - Table 2: Heterophilic vertex classification (Chameleon, Squirrel, Actor, Minesweeper, Roman-Empire)
# - Table 3: Graph-level regression (ZINC)
# - Baselines: GCN and MLP on selected datasets
# - Table 5: Ablation study on Cora and Roman-Empire
#
# Usage:
#   bash reproduce.sh          # Full reproduction (several hours)
#   bash reproduce.sh --quick  # Quick mode with fewer seeds (~30 min)
#
# Results are saved to /workspace/results/

set -e

cd /workspace

# Install dependencies if needed
pip install torch-scatter torch-sparse torch-geometric -q 2>/dev/null || true

# Create results directory
mkdir -p results

echo "============================================================"
echo "HyperAggregation Paper Replication"
echo "============================================================"

# Check for --quick flag
QUICK_FLAG=""
if [ "$1" == "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "Running in QUICK mode (fewer seeds, shorter training)"
fi

# ============================================================
# Step 1: Run main experiments (Tables 1, 2, 3 + baselines)
# ============================================================
echo ""
echo "Step 1: Running main experiments (Tables 1, 2, 3 + baselines)..."
echo "============================================================"
python run_reproduce.py $QUICK_FLAG --skip-existing --baselines

# ============================================================
# Step 2: Run ablation study (Table 5)
# ============================================================
echo ""
echo "Step 2: Running ablation study (Table 5)..."
echo "============================================================"
if [ -f results/ablation_table5.json ]; then
    echo "  Ablation results already exist, skipping..."
else
    python run_ablation_fast.py
fi

# ============================================================
# Step 3: Generate result tables
# ============================================================
echo ""
echo "Step 3: Generating result tables..."
echo "============================================================"
python generate_tables.py

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Results saved to /workspace/results/"
echo "Summary table: /workspace/results/results_tables.md"
echo "============================================================"
cat results/results_tables.md
