#!/usr/bin/env bash
# run_ns3.sh -- JCC revision 2: runs every allocation x capacity x network seed in parallel.
#
# usage:  ./run_ns3.sh <preset>        presets: smoke | main | aligned | density | list
#   smoke    7 users (one per app), Observed + Persistence, caps 0/10, direct/lb/lbres, 1 run -> minutes
#   main     trace timing ("real"), caps 0(unlimited) 40 10 5 1 Mbps
#   aligned  all users' test periods start together (up to 83 concurrent users), caps 0 100 50 25
#   density  aligned scenario with every user replicated x2 and x4, cap 50
#   list     print the number of jobs / ns-3 runs of every preset without running anything
#
# Revision-2 design (Reviewer 1: same repeated simulation seeds for all policies, training vs
# network variability separated):
#   * every allocator runs on the SAME network seeds (ns-3 RngRun) 1..RUNS
#   * learned models: training seed s is simulated with network seed s+1 (seeds 0..RUNS-1), so
#     every network seed carries one training seed of every model (paired with the baselines)
#   * CROSSED models (default GRU): every training seed x every network seed in main and aligned,
#     which separates training variability from network variability
#   * residual-aware LB (lbres / leastloadres) for RES_MODELS + LB-only, as a sensitivity analysis
#
# environment (all optional):
#   NS3_DIR    ns-3 tree that contains scratch/eua_trace_sim_r2.cc   (default: ~/ns-3-dev)
#   JOBS       parallel simulations (default: number of cores - 1)
#   RUNS       network seeds per configuration           (default 3)
#   SEEDS      training seeds of the learned models used  (default "0 1 2")
#   CROSSED    learned models simulated with every seed x every run in main/aligned (default "GRU")
#   RES_MODELS allocators that also run the residual-aware LB (default "Observed Persistence GRU")
#   GROUP      all | baselines | learned                  (default all)
#   CAPS       override the capacity list of the preset, e.g. CAPS="0 10"
#   MODELS     restrict to these allocators, e.g. MODELS="Observed GRU Persistence"
#   DENSITY_LEARNED  learned models simulated in the density preset (default: all 11), e.g. "GRU LSTM"
#   EXTRA      extra arguments passed to eua_trace_sim_r2 (e.g. EXTRA="--qdisc=codel")
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-}"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
RUNS="${RUNS:-3}"
SEEDS="${SEEDS:-0 1 2}"
CROSSED="${CROSSED-GRU}"
RES_MODELS="${RES_MODELS-Observed Persistence GRU}"
GROUP="${GROUP:-all}"
EXTRA="${EXTRA:-}"
INPUTS="$HERE/ns3_inputs"
RES="$HERE/results_ns3"

DL_MODELS="GRU LSTM DNN Transformer Autoformer TFT FEDformer DeepCog Dong DeepSlice Chen"
BASELINES="Observed Persistence UserMajority AppMajority NearestServer"
NO_LBONLY=0
case "$GROUP" in
  baselines) DL_MODELS="" ;;
  learned)   BASELINES=""; NO_LBONLY=1 ;;
  all)       ;;
  *) echo "GROUP must be all, baselines or learned"; exit 1 ;;
esac

usage() { sed -n '2,33p' "$0"; exit 1; }
[[ -z "$PRESET" ]] && usage

want() {  # is allocator $1 selected by MODELS?
  [[ -z "${MODELS:-}" ]] && return 0
  for m in $MODELS; do [[ "$m" == "$1" ]] && return 0; done
  return 1
}
inlist() { local x="$1"; shift; for y in "$@"; do [[ "$x" == "$y" ]] && return 0; done; return 1; }
allruns() { seq -s, 1 "$RUNS"; }

# ---------------------------------------------------------------- job list
# line format: scenario density model decisions policy cap runs(comma list) subset
emit() { echo "$1 $2 $3 $4 $5 $6 $7 ${8:-0}"; }

jobs_for() {
  local scen="$1" dens="$2" caps="$3" policies="$4" subset="${5:-0}"
  local cap pol b m s runs
  for cap in $caps; do
    for pol in $policies; do
      # unlimited capacity: load balancing never triggers -> only 'direct'
      if [[ "$cap" == "0" && "$pol" != "direct" ]]; then continue; fi
      if [[ "$pol" == "leastload" ]]; then
        if [[ "$NO_LBONLY" != "1" ]] && want LBOnly; then
          emit "$scen" "$dens" LBOnly Persistence__s0 leastload "$cap" "$(allruns)" "$subset"
          emit "$scen" "$dens" LBOnly Persistence__s0 leastloadres "$cap" "$(allruns)" "$subset"
        fi
        continue
      fi
      for b in $BASELINES; do
        [[ "$pol" == "lbk" ]] && continue          # baselines have no meaningful ranked list
        want "$b" || continue
        emit "$scen" "$dens" "$b" "${b}__s0" "$pol" "$cap" "$(allruns)" "$subset"
        if [[ "$pol" == "lb" ]] && inlist "$b" $RES_MODELS; then
          emit "$scen" "$dens" "$b" "${b}__s0" lbres "$cap" "$(allruns)" "$subset"
        fi
      done
      for m in $DL_MODELS; do
        want "$m" || continue
        for s in $SEEDS; do
          if inlist "$m" $CROSSED && [[ "$dens" == "1" ]]; then
            runs="$(allruns)"                          # crossed: every network seed
          else
            runs="$(( s % RUNS + 1 ))"                 # paired: training seed s <-> network seed s+1
          fi
          emit "$scen" "$dens" "$m" "${m}__s$s" "$pol" "$cap" "$runs" "$subset"
          if [[ "$pol" == "lb" ]] && inlist "$m" $RES_MODELS; then
            emit "$scen" "$dens" "$m" "${m}__s$s" lbres "$cap" "$runs" "$subset"
          fi
        done
      done
    done
  done
}

make_jobs() {
  case "$1" in
    smoke)   GROUP=all MODELS="Observed Persistence" RUNS=1 SEEDS=0 CROSSED="" RES_MODELS="Observed Persistence" \
               jobs_for aligned 1 "${CAPS:-0 10}" "direct lb" 7 ;;
    main)    jobs_for real 1 "${CAPS:-0 40 10 5 1}" "direct lb leastload" ;;
    aligned) jobs_for aligned 1 "${CAPS:-0 100 50 25}" "direct lb lbk leastload" ;;
    density) [[ -n "$DL_MODELS" && -n "${DENSITY_LEARNED:-}" ]] && DL_MODELS="$DENSITY_LEARNED"
             for d in 2 4; do jobs_for aligned $d "${CAPS:-50}" "direct lb leastload"; done ;;
    *) usage ;;
  esac
}

nruns() { awk '{n+=split($7,a,",")} END{print n+0}'; }

if [[ "$PRESET" == "list" ]]; then
  for p in main aligned density; do
    n=$(make_jobs $p | wc -l); s=$(make_jobs $p | nruns)
    echo "$p: $n allocation jobs, $s ns-3 runs"
  done
  exit 0
fi

# ---------------------------------------------------------------- binary
BIN="$(find "$NS3_DIR/build" -type f -name '*eua_trace_sim_r2*' -perm -u+x 2>/dev/null | head -n1 || true)"
if [[ -z "$BIN" ]]; then
  echo "eua_trace_sim_r2 not built. Copy eua_trace_sim_r2.cc to $NS3_DIR/scratch/ and run:"
  echo "   cd $NS3_DIR && ./ns3 build scratch/eua_trace_sim_r2"
  exit 1
fi
export LD_LIBRARY_PATH="$NS3_DIR/build/lib:${LD_LIBRARY_PATH:-}"
echo "binary : $BIN"
[[ "$BIN" == *optimized* || "$BIN" == *release* ]] || \
  echo "NOTE   : this looks like a debug/default build; an optimized build is several times faster."

JOBFILE="$RES/jobs_${PRESET}${MODELS:+_${MODELS// /-}}.txt"
mkdir -p "$RES"
make_jobs "$PRESET" > "$JOBFILE"
NJ=$(wc -l < "$JOBFILE"); NR=$(nruns < "$JOBFILE")
echo "preset : $PRESET -> $NJ allocation jobs, $NR ns-3 runs, $JOBS in parallel"
echo "results: $RES"

worker() {
  local scen=$1 dens=$2 model=$3 dec=$4 pol=$5 cap=$6 runs=$7 subset=$8
  local seedtag="${dec##*__}"
  local tag="${scen}__d${dens}__${model}__${seedtag}__${pol}__cap${cap}"
  [[ "$subset" != "0" ]] && tag="smoke_${tag}"
  local dir="$RES/$scen"
  mkdir -p "$dir/net" "$dir/srv" "$dir/alloc" "$dir/logs" "$dir/tmp"
  local todo=0 r
  for r in ${runs//,/ }; do [[ -s "$dir/net/${tag}__run${r}.csv" ]] || todo=1; done
  [[ $todo == 0 ]] && { echo "skip $tag (done)"; return 0; }
  local flows="$dir/tmp/${tag}.csv" checks="$dir/tmp/${tag}_checks.csv"
  local alloc_chk="" sim_chk=""
  if [[ "$pol" != "direct" && "$cap" != "0" ]]; then
    alloc_chk="--checks $checks"; sim_chk="--checks=$checks"
  fi
  python3 "$HERE/allocate.py" --inputs "$INPUTS" --decisions "$dec" --policy "$pol" --cap "$cap" \
      --scenario "$scen" --density "$dens" --user-subset "$subset" \
      --out "$flows" --stats "$dir/alloc/${tag}.json" $alloc_chk > /dev/null
  for r in ${runs//,/ }; do
    local out="$dir/net/${tag}__run${r}.csv" srv="$dir/srv/${tag}__run${r}.csv"
    [[ -s "$out" ]] && continue
    if "$BIN" --flows="$flows" --cap="$cap" --run="$r" --out="$out.part" --outSrv="$srv.part" \
         $sim_chk --verbose=false $EXTRA > "$dir/logs/${tag}__run${r}.log" 2>&1; then
      mv "$srv.part" "$srv"; mv "$out.part" "$out"; tail -n1 "$dir/logs/${tag}__run${r}.log"
    else
      echo "FAILED $tag run $r -- see $dir/logs/${tag}__run${r}.log"
    fi
  done
  rm -f "$flows" "$checks"
}
export -f worker
export RES INPUTS HERE BIN EXTRA

T0=$(date +%s)
xargs -P "$JOBS" -L 1 bash -c 'worker "$@"' _ < "$JOBFILE"
echo "finished in $(( ($(date +%s) - T0) / 60 )) min."
