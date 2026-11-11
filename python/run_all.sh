#!/usr/bin/env bash
# SS5G v3.0 -- whole Python pipeline (Linux/macOS). Run inside this python folder with your
# TensorFlow environment active:   ./run_all.sh
# Put the two GAViST5G CSV files in ../data first (see ../data/README.md).
# Every step can be re-run; finished tuning runs, models and seeds are skipped.
#
#   USE_RELEASED_TUNING=1 ./run_all.sh   skips the Bayesian optimisation and uses the selected
#                                        values in tuning/ (the values reported in the paper)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DATA=../data/CombinedALLGeo.csv
GLOBAL=../data/YouTubeCombinedGeo.csv
SEEDS="0 1 2 3 4"

echo "---- 1. prepare data"
python eua_pipeline.py prepare --data "$DATA" --work work
python eua_pipeline.py prepare --data "$GLOBAL" --single-app YouTube --work work_global

echo "---- 2. smoke tests (7 models, 4 methods from the literature)"
python eua_pipeline.py smoke --work work --out results
python sota_models.py smoke --work work --out results

echo "---- 3. baselines"
python eua_pipeline.py baselines --work work --out results
python eua_pipeline.py baselines --work work_global --out results_global

echo "---- 4. hyperparameter tuning (Bayesian optimisation, 15 trials per model)"
if [[ "${USE_RELEASED_TUNING:-0}" == "1" ]]; then
  mkdir -p results/tuning && cp tuning/*.json results/tuning/
else
  python eua_pipeline.py tune --work work --out results --trials 15
  python sota_models.py tune --work work --out results --trials 15
fi

echo "---- 5. training with the selected values: 11 models x 5 seeds"
python eua_pipeline.py train --work work --out results --use-tuned --feature-sets full --seeds $SEEDS
python sota_models.py train --work work --out results --use-tuned --feature-sets full --seeds $SEEDS

echo "---- 6. inputs of the ns-3 evaluation (../ns3/ns3_inputs)"
python ../ns3/make_ns3_inputs.py --work work --results results --out ../ns3/ns3_inputs

echo "---- 7. architecture tables (layers, shapes, parameters, selected values)"
python eua_pipeline.py describe --work work --out results --use-tuned
python sota_models.py describe --work work --out results --use-tuned

echo "---- 8. feature ablation, unseen users, international portion"
python eua_pipeline.py train --work work --out results --use-tuned --models DNN GRU Transformer FEDformer \
  --feature-sets no_rtt no_srv --seeds $SEEDS
python eua_pipeline.py holdout --work work --out results --use-tuned --models GRU DNN
python eua_pipeline.py train --work work_global --out results_global --use-tuned --tuning-dir results/tuning \
  --models all --feature-sets full --seeds 0 1 2

echo "---- 9. inference latency: GPU, then CPU only"
python eua_pipeline.py latency --work work --out results --use-tuned
python sota_models.py latency --work work --out results --use-tuned
CUDA_VISIBLE_DEVICES=-1 python eua_pipeline.py latency --work work --out results --use-tuned
CUDA_VISIBLE_DEVICES=-1 python sota_models.py latency --work work --out results --use-tuned

echo "---- 10. reports"
python eua_pipeline.py report --out results
python eua_pipeline.py report --out results_global

echo "DONE. Open results/REPORT.md; the ns-3 inputs are in ../ns3/ns3_inputs"
