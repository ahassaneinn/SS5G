#!/usr/bin/env bash
# run_everything.sh -- SS5G v3.0: the whole ns-3 study in one command.
#   build eua_trace_sim_r2 -> smoke test -> main + aligned (all allocators) -> density
# Resumable: run it again after an interruption and finished runs are skipped.
#
#   cd ~/ss5g/ns3 && NS3_DIR=~/ns-3-dev nohup ./run_everything.sh > run_everything.out 2>&1 &
#   (optional, ~30 h shorter) DENSITY_LEARNED="GRU LSTM" ./run_everything.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
export JOBS="${JOBS:-7}"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- inputs present?
for f in steps.csv.gz traffic.csv.gz servers.csv users.csv; do
  [[ -s ns3_inputs/$f ]] || { say "missing ns3_inputs/$f -- run ../python/run_all first (it writes ns3_inputs/)"; exit 1; }
done
n=$(ls ns3_inputs/decisions/*.csv.gz 2>/dev/null | wc -l)
[[ $n -ge 60 ]] || { say "only $n decision files in ns3_inputs/decisions (expected 60)"; exit 1; }

# ---- build eua_trace_sim_r2 in the ns-3 tree
say "building eua_trace_sim_r2 in $NS3_DIR"
cp eua_trace_sim_r2.cc "$NS3_DIR/scratch/" || exit 1
( cd "$NS3_DIR" && ./ns3 build scratch/eua_trace_sim_r2 ) || { say "BUILD FAILED -- see the lines above"; exit 1; }

# ---- smoke test
say "smoke test"
bash ./run_ns3.sh smoke | tee smoke.out
if grep -q FAILED smoke.out || [[ $(grep -c '^DONE' smoke.out) -lt 8 ]]; then
  say "SMOKE TEST FAILED -- see smoke.out and results_ns3/aligned/logs/smoke_*"; exit 1
fi

# ---- main + aligned, then density
say "main + aligned (all allocators)"
SCENARIOS="main aligned" GROUP=all bash ./run_all_models.sh
say "density${DENSITY_LEARNED:+ (learned models: $DENSITY_LEARNED)}"
SCENARIOS="density" GROUP=all bash ./run_all_models.sh
say "ALL SIMULATIONS DONE -- summaries in results_ns3/by_model/"
