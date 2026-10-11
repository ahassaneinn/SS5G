#!/usr/bin/env bash
# run_all_models.sh -- ONE command for the whole ns-3 study.
#
#   NS3_DIR=~/ns-3-dev ./run_all_models.sh            # run everything (resumable)
#   ./run_all_models.sh status                         # progress table (any time, from another terminal)
#
# Order: scenario by scenario (main -> aligned -> density); inside each scenario the 11 DL models
# one after another (DNN, LSTM, GRU, Transformer, Autoformer, TFT, FEDformer, DeepCog, Dong,
# DeepSlice, Chen), then the reference allocators (Observed, Persistence, LBOnly, UserMajority,
# AppMajority, NearestServer).  Each model/scenario combination runs all its seeds,
# capacities, policies and replications in parallel, then writes its own files:
#
#   results_ns3/by_model/<scenario>/<Model>_summary.md    readable tables (open while running)
#   results_ns3/by_model/<scenario>/<Model>_summary.csv   mean / sd / 95% CI per policy x capacity
#   results_ns3/by_model/<scenario>/<Model>_runs.csv      every finished run
#
# The summary of the combination that is currently running is refreshed every REFRESH seconds,
# so you can watch it fill in.  Stop with Ctrl+C at any time; the same command resumes and skips
# finished runs.
#
# environment (all optional):
#   NS3_DIR     ns-3 tree with scratch/eua_trace_sim.cc built   (default ~/ns-3-dev)
#   JOBS        parallel simulations                           (default: cores - 1)
#   RUNS_DL     replications per DL model seed                 (default 4 -> 20 runs per model)
#   RUNS_BASE   replications per deterministic baseline        (default 20)
#   SCENARIOS   default "main aligned density"
#   MODELS      default "DNN LSTM GRU Transformer Autoformer TFT FEDformer DeepCog Dong DeepSlice Chen"
#   REFERENCE   reference allocators; default "Observed Persistence LBOnly UserMajority AppMajority
#               NearestServer"  (REFERENCE="" -> only the DL models)
#   REFRESH     seconds between live summary refreshes        (default 300)
#   EXTRA       extra flags for eua_trace_sim (e.g. "--qdisc=codel")
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RES="$HERE/results_ns3"
SCENARIOS="${SCENARIOS:-main aligned density}"
MODELS_ALL="${MODELS:-DNN LSTM GRU Transformer Autoformer TFT FEDformer DeepCog Dong DeepSlice Chen}"
REFERENCE="${REFERENCE-Observed Persistence LBOnly UserMajority AppMajority NearestServer}"
REFRESH="${REFRESH:-300}"
export NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
mkdir -p "$RES"
LOG="$RES/run_all.log"

# ------------------------------------------------------------------ status
status() {
  python3 - "$RES" "$SCENARIOS" "$MODELS_ALL $REFERENCE" <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
from summarize_combo import summarize, SCEN
res, scens, models = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
print(f"{'scenario':<9} {'model':<14} {'done':>6} {'planned':>8}   summary file")
for s in scens:
    for m in models:
        done, exp = summarize(res, s, m, quiet=True)
        if done == 0 and exp is None:
            state = "not started"
        else:
            state = f"{res}/by_model/{s}/{m}_summary.md"
        print(f"{s:<9} {m:<14} {done:>6} {exp if exp is not None else '?':>8}   {state}")
PY
}
if [[ "${1:-}" == "status" ]]; then status; exit 0; fi

# ------------------------------------------------------------------ checks
if ! find "$NS3_DIR/build" -type f -name '*eua_trace_sim*' -perm -u+x 2>/dev/null | grep -q .; then
  echo "eua_trace_sim is not built in $NS3_DIR. Do this first:"
  echo "   cp eua_trace_sim.cc $NS3_DIR/scratch/ && cd $NS3_DIR && ./ns3 build scratch/eua_trace_sim"
  exit 1
fi
[[ -x "$HERE/run_ns3.sh" ]] || chmod +x "$HERE/run_ns3.sh"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# planned work, printed before starting
TOTAL=0
for s in $SCENARIOS; do
  for m in $MODELS_ALL $REFERENCE; do
    n=$(MODELS="$m" "$HERE/run_ns3.sh" list 2>/dev/null | awk -v p="$s:" '$1==p {print $(NF-2)}')
    TOTAL=$(( TOTAL + ${n:-0} ))
  done
done
say "START  scenarios: $SCENARIOS | models: $MODELS_ALL | reference: ${REFERENCE:-none} | planned ns-3 runs: $TOTAL"
say "live summaries: $RES/by_model/<scenario>/<Model>_summary.md   (progress: ./run_all_models.sh status)"

REFRESHER=""
cleanup() { [[ -n "$REFRESHER" ]] && kill "$REFRESHER" 2>/dev/null; }
trap 'cleanup; say "INTERRUPTED -- run the same command again to resume"; exit 130' INT TERM

T_ALL=$(date +%s)
for s in $SCENARIOS; do
  for m in $MODELS_ALL $REFERENCE; do
    T0=$(date +%s)
    say "BEGIN  $s / $m"
    # live refresh of this combination's summary while it runs
    ( while sleep "$REFRESH"; do python3 "$HERE/summarize_combo.py" --results "$RES" --scenario "$s" \
        --model "$m" --quiet 2>/dev/null; done ) &
    REFRESHER=$!
    MODELS="$m" "$HERE/run_ns3.sh" "$s" >> "$RES/run_ns3_${s}_${m}.log" 2>&1
    rc=$?
    kill "$REFRESHER" 2>/dev/null; wait "$REFRESHER" 2>/dev/null; REFRESHER=""
    python3 "$HERE/summarize_combo.py" --results "$RES" --scenario "$s" --model "$m" | tee -a "$LOG"
    fails=$(grep -c '^FAILED' "$RES/run_ns3_${s}_${m}.log" 2>/dev/null || true)
    [[ $rc -ne 0 || ${fails:-0} -gt 0 ]] && say "WARN   $s / $m: exit code $rc, $fails failed runs -- see $RES/run_ns3_${s}_${m}.log"
    say "END    $s / $m  ($(( ($(date +%s) - T0) / 60 )) min)"
  done
done
say "ALL DONE in $(( ($(date +%s) - T_ALL) / 3600 )) h $(( ($(date +%s) - T_ALL) % 3600 / 60 )) min"

# combined report (needs pandas, scipy, matplotlib; otherwise run analyze_ns3.py on Windows)
LAT="${LATENCY_CPU:-}"
if python3 -c "import pandas, scipy, matplotlib" 2>/dev/null; then
  python3 "$HERE/analyze_ns3.py" --results "$RES" ${LAT:+--latency-cpu "$LAT"} | tee -a "$LOG"
  say "combined report: $RES/report/REPORT_ns3.md"
else
  say "pandas/scipy/matplotlib not found: run analyze_ns3.py on Windows on the results_ns3 folder"
fi
