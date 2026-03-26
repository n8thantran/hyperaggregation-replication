#!/bin/bash
# Reproduce key results from "HyperAggregation: Aggregating over Graph Edges with Hypernetworks"
#
# This script runs the main GHC experiments from Tables 1, 2, and 3 of the paper,
# plus selected baselines (GCN, MLP).
#
# Results are saved to /workspace/results/ as JSON files and a summary table.
#
# Usage:
#   bash reproduce.sh          # Full reproduction (all tables, all seeds)
#   bash reproduce.sh --quick  # Quick mode (fewer seeds, shorter training)

set -e

cd /workspace

# Install dependencies if needed
pip install torch-scatter torch-sparse torch-geometric numpy -q 2>/dev/null || true

echo "============================================"
echo "HyperAggregation Paper Replication"
echo "============================================"
echo ""

# Create results directory
mkdir -p results

# Run all experiments
# --skip-existing allows resuming if interrupted
python run_reproduce.py --skip-existing "$@"

echo ""
echo "============================================"
echo "Results saved to /workspace/results/"
echo "Summary table: /workspace/results/results_tables.md"
echo "============================================"
cat /workspace/results/results_tables.md
