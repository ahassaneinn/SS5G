#!/usr/bin/env bash
# Full Python pipeline (Linux/macOS). Run inside this python folder with your
# TensorFlow environment active:   ./run_all.sh
# Put the two GAViST5G CSV files in ../data first (see ../data/README.md).
# Every stage writes to results/ (results_global/ for the international portion)
# and can be re-run; finished model/seed combinations are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DATA=../data/dataset1.csv
GLOBAL=../data/dataset2.csv
SEEDS="0 1 2 3 4"

echo "---- 1. prepare data"
python eua_pipeline.py prepare --data "$DATA" --work work
python eua_pipeline.py prepare --data "$GLOBAL" --single-app YouTube --work work_global

echo "---- 2. smoke test"
python eua_pipeline.py smoke --work work --out results

echo "---- 3. baselines"
python eua_pipeline.py baselines --work work --out results
python eua_pipeline.py baselines --work work_global --out results_global

# Optional: equal-budget Bayesian optimisation (several hours). Then add
# --use-tuned to the train and latency commands below.
# python eua_pipeline.py tune --work work --out results --models all --trials 15

echo "---- 4. seven DL models x 5 seeds"
python eua_pipeline.py train --work work --out results --models all --feature-sets full --seeds $SEEDS

echo "---- 5. feature ablation: no RTT, no server history"
python eua_pipeline.py train --work work --out results --models DNN GRU Transformer FEDformer \
  --feature-sets no_rtt no_srv --seeds $SEEDS

echo "---- 6. unseen users: 5 user-disjoint folds"
python eua_pipeline.py holdout --work work --out results --models GRU DNN

echo "---- 7. international portion"
python eua_pipeline.py train --work work_global --out results_global --models all --feature-sets full --seeds 0 1 2

echo "---- 8. inference latency: GPU, then CPU only"
python eua_pipeline.py latency --work work --out results
CUDA_VISIBLE_DEVICES=-1 python eua_pipeline.py latency --work work --out results

echo "---- 9. reports"
python eua_pipeline.py report --out results
python eua_pipeline.py report --out results_global

echo "---- 10. inputs of the ns-3 evaluation (../ns3/ns3_inputs)"
python ../ns3/make_ns3_inputs.py --python-dir . --out ../ns3/ns3_inputs

echo "DONE. Open results/REPORT.md"
