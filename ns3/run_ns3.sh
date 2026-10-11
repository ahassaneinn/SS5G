#!/usr/bin/env bash
# run_ns3.sh -- runs every allocation x capacity x replication in parallel.
#
# usage:  ./run_ns3.sh <preset>        presets: smoke | main | aligned | density | list
#   smoke    7 users (one per app), 2 allocators, 2 caps, 1 run          -> a couple of minutes
#   main     trace timing ("real"), caps 0(unlimited) 40 10 5 1 Mbps
#   aligned  all users' test periods start together (up to 83 concurrent users), caps 0 100 50 25
#   density  aligned scenario with every user replicated x2 and x4 (scalability), cap 50
#   list     print the job list of all presets without running anything
#
# environment (all optional):
#   NS3_DIR    path of the ns-3 tree that contains scratch/eua_trace_sim.cc  (default: ~/ns-3-dev)
#   JOBS       parallel simulations (default: number of cores - 1)
#   RUNS_DL    ns-3 replications per DL model seed   (default 4 -> 5 seeds x 4 = 20 runs per model)
#   RUNS_BASE  ns-3 replications per deterministic baseline (default 20)
#   CAPS       override the capacity list of the preset, e.g. CAPS="0 10"
#   MODELS     restrict to these allocators, e.g. MODELS="Observed GRU Persistence"
#   EXTRA      extra arguments passed to eua_trace_sim (e.g. EXTRA="--qdisc=codel")
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-}"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
RUNS_DL="${RUNS_DL:-4}"
RUNS_BASE="${RUNS_BASE:-20}"
EXTRA="${EXTRA:-}"
INPUTS="$HERE/ns3_inputs"
RES="$HERE/results_ns3"

DL_MODELS="DNN LSTM GRU Transformer Autoformer TFT FEDformer DeepCog Dong DeepSlice Chen"
BASELINES="Observed Persistence UserMajority AppMajority NearestServer"

usage() { sed -n '2,20p' "$0"; exit 1; }
[[ -z "$PRESET" ]] && usage

want() {  # is allocator $1 selected by MODELS?
  [[ -z "${MODELS:-}" ]] && return 0
  for m in $MODELS; do [[ "$m" == "$1" ]] && return 0; done
  return 1
}

# ---------------------------------------------------------------- job list
# line format: scenario density model decisions policy cap runs subset
emit() { echo "$1 $2 $3 $4 $5 $6 $7 ${8:-0}"; }

jobs_for() {
  local scen="$1" dens="$2" caps="$3" policies="$4" subset="${5:-0}" rdl="${6:-$RUNS_DL}" rbase="${7:-$RUNS_BASE}"
  for cap in $caps; do
    for pol in $policies; do
      # unlimited capacity: load balancing never triggers -> only 'direct'
      if [[ "$cap" == "0" && "$pol" != "direct" ]]; then continue; fi
      if [[ "$pol" == "leastload" ]]; then
        want LBOnly && emit "$scen" "$dens" LBOnly Persistence__s0 leastload "$cap" "$rbase" "$subset"
        continue
      fi
      for b in $BASELINES; do
        [[ "$pol" == "lbk" ]] && continue          # baselines have no meaningful ranked list
        want "$b" && emit "$scen" "$dens" "$b" "${b}__s0" "$pol" "$cap" "$rbase" "$subset"
      done
      for m in $DL_MODELS; do
        want "$m" || continue
        for s in 0 1 2 3 4; do emit "$scen" "$dens" "$m" "${m}__s$s" "$pol" "$cap" "$rdl" "$subset"; done
      done
    done
  done
}

make_jobs() {
  case "$1" in
    smoke)   MODELS="Observed GRU" jobs_for aligned 1 "${CAPS:-0 10}" "direct lb" 7 1 1 | awk '$4 ~ /__s0$/' ;;
    main)    jobs_for real 1 "${CAPS:-0 40 10 5 1}" "direct lb leastload" ;;
    aligned) jobs_for aligned 1 "${CAPS:-0 100 50 25}" "direct lb lbk leastload" ;;
    density) for d in 2 4; do MODELS="${MODELS:-Observed Persistence LBOnly GRU LSTM}" \
               jobs_for aligned $d "${CAPS:-50}" "direct lb leastload" 0 2 10; done ;;
    *) usage ;;
  esac
}

if [[ "$PRESET" == "list" ]]; then
  for p in main aligned density; do
    n=$(make_jobs $p | wc -l); s=$(make_jobs $p | awk '{s+=$7} END{print s}')
    echo "$p: $n allocation jobs, $s ns-3 runs"
  done
  exit 0
fi

# ---------------------------------------------------------------- binary
BIN="$(find "$NS3_DIR/build" -type f -name '*eua_trace_sim*' -perm -u+x 2>/dev/null | head -n1 || true)"
if [[ -z "$BIN" ]]; then
  echo "eua_trace_sim not built. Copy eua_trace_sim.cc to $NS3_DIR/scratch/ and run:"
  echo "   cd $NS3_DIR && ./ns3 build scratch/eua_trace_sim"
  exit 1
fi
export LD_LIBRARY_PATH="$NS3_DIR/build/lib:${LD_LIBRARY_PATH:-}"
echo "binary : $BIN"
[[ "$BIN" == *optimized* || "$BIN" == *release* ]] || \
  echo "NOTE   : this looks like a debug/default build; an optimized build is several times faster (see README)."

JOBFILE="$RES/jobs_${PRESET}${MODELS:+_${MODELS// /-}}.txt"
mkdir -p "$RES"
make_jobs "$PRESET" > "$JOBFILE"
NJ=$(wc -l < "$JOBFILE"); NR=$(awk '{s+=$7} END{print s}' "$JOBFILE")
echo "preset : $PRESET -> $NJ allocation jobs, $NR ns-3 runs, $JOBS in parallel"
echo "results: $RES"

worker() {
  local scen=$1 dens=$2 model=$3 dec=$4 pol=$5 cap=$6 runs=$7 subset=$8
  local seedtag="${dec##*__}"
  local tag="${scen}__d${dens}__${model}__${seedtag}__${pol}__cap${cap}"
  [[ "$subset" != "0" ]] && tag="smoke_${tag}"
  local dir="$RES/$scen"
  mkdir -p "$dir/net" "$dir/alloc" "$dir/logs" "$dir/tmp"
  local todo=0
  for r in $(seq 1 "$runs"); do [[ -s "$dir/net/${tag}__run${r}.csv" ]] || todo=1; done
  [[ $todo == 0 ]] && { echo "skip $tag (done)"; return 0; }
  local flows="$dir/tmp/${tag}.csv"
  python3 "$HERE/allocate.py" --inputs "$INPUTS" --decisions "$dec" --policy "$pol" --cap "$cap" \
      --scenario "$scen" --density "$dens" --user-subset "$subset" \
      --out "$flows" --stats "$dir/alloc/${tag}.json" > /dev/null
  for r in $(seq 1 "$runs"); do
    local out="$dir/net/${tag}__run${r}.csv"
    [[ -s "$out" ]] && continue
    if "$BIN" --flows="$flows" --cap="$cap" --run="$r" --out="$out.part" --verbose=false $EXTRA \
         > "$dir/logs/${tag}__run${r}.log" 2>&1; then
      mv "$out.part" "$out"; tail -n1 "$dir/logs/${tag}__run${r}.log"
    else
      echo "FAILED $tag run $r -- see $dir/logs/${tag}__run${r}.log"
    fi
  done
  rm -f "$flows"
}
export -f worker
export RES INPUTS HERE BIN EXTRA

T0=$(date +%s)
xargs -P "$JOBS" -L 1 bash -c 'worker "$@"' _ < "$JOBFILE"
echo "finished in $(( ($(date +%s) - T0) / 60 )) min. Next: python3 analyze_ns3.py --results $RES"
