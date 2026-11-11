# Trace-driven ns-3 evaluation (SS5G v3.0)

These scripts replay the server assignments predicted in `../python` in an edge-server-level, trace-driven ns-3 simulation. Every allocator gets exactly the same traffic.

| File | What it does |
|---|---|
| `make_ns3_inputs.py` | Builds `ns3_inputs/` (test-period traffic, servers, users, ranked decisions of every model and baseline) from `../python/work` and `../python/results`. `run_all` in `../python` calls it. |
| `allocate.py` | Allocation rules (standard library only): `direct`, `lb`, `lbk` (LB with the model's top-5 candidates), `leastload` (LB without prediction), and the residual-aware `lbres` / `leastloadres`. Logs every capacity check and the allocation metrics. |
| `eua_trace_sim_r2.cc` | ns-3 program that replays a flow schedule with common random numbers and measures QoS, server utilisation, overload, residual traffic, false admissions and reassignments |
| `run_ns3.sh` | Runs one preset (`smoke`, `main`, `aligned`, `density`, `list`) in parallel |
| `run_all_models.sh` | All allocators and scenarios; `./run_all_models.sh status` shows the progress |
| `run_everything.sh` | Build, smoke test, then the whole study in one command (resumable) |
| `summarize_combo.py` | Per model/scenario summaries (safe to run while simulations are running) |
| `analyze_ns3.py` | Summary tables with 95% CIs and Welch tests |
| `make_paper_results.py` | Tables 8–13 and Figures 1–6 of the article (`paper_results/`) |

## Design

- **Network seeds.** Every allocator runs with the same network seeds 1–3 (`RUNS`), which set the access delays and packet timing; within a run, every allocator uses the same random numbers.
- **Training seeds.** For the learned models, run *r* uses the model trained with seed *r*−1. GRU also runs every training seed × network seed combination in `main` and `aligned` (`CROSSED`), which separates training from network variability.
- **Residual-aware LB.** `lbres` and `leastloadres` add an estimate of each server's queued and draining traffic to the capacity check. They run for `RES_MODELS` (default: Observed, Persistence, GRU) and for LB without prediction.
- **Scenarios.** `main`: trace timing, capacities unlimited/40/10/5/1 Mbps. `aligned`: all users' test periods start together, 100/50/25 Mbps. `density`: every user replicated ×2 and ×4 at 50 Mbps.

## 1. Set up (once)

You need ns-3.36 or newer (CMake-based `./ns3` script) and Python 3 with pandas, scipy and matplotlib for the analysis.

```bash
# copy this folder (with ns3_inputs/ from ../python/run_all) to the Linux machine, e.g. ~/ss5g/ns3
cd ~/ss5g/ns3 && chmod +x *.sh
# on "bad interpreter" or "\r" errors (Windows line endings):  sed -i 's/\r$//' *.sh *.py
cd ~/ns-3-dev        # your ns-3 folder
./ns3 configure --build-profile=optimized --disable-examples --disable-tests   # recommended, much faster
```

## 2. Run everything

```bash
cd ~/ss5g/ns3
NS3_DIR=~/ns-3-dev JOBS=7 nohup ./run_everything.sh > run_everything.out 2>&1 &
./run_all_models.sh status          # progress, at any time
```

`run_everything.sh` copies and builds `eua_trace_sim_r2`, runs a smoke test, then `main` and `aligned` for all allocators, then `density`. Run it again after an interruption: finished runs are skipped. `./run_ns3.sh list` prints the number of planned runs of each preset.

To run one simulation by hand:

```bash
python3 allocate.py --decisions GRU__s0 --policy lbres --cap 10 --scenario real --out /tmp/f.csv --stats /tmp/a.json --checks /tmp/c.csv
cd ~/ns-3-dev && ./ns3 run "eua_trace_sim_r2 --flows=/tmp/f.csv --cap=10 --run=1 --out=/tmp/r.csv --outSrv=/tmp/s.csv --checks=/tmp/c.csv"
```

## 3. Outputs

- `results_ns3/<scenario>/net/*.csv`: QoS per run (goodput, delivered traffic, loss, delay, jitter, stall proxy), for all traffic and per application;
- `results_ns3/<scenario>/srv/*.csv`: server-level metrics per run (utilisation variance, overloaded user-seconds, residual traffic, false admissions, reassignments);
- `results_ns3/<scenario>/alloc/*.json`: allocation metrics (LB override rate, reassignment rates, planned load);
- `results_ns3/by_model/<scenario>/<Model>_{summary.md,summary.csv,runs.csv,srv_runs.csv}`.

The trace-timing scenario is stored in `results_ns3/real/`; the replicated runs are in `results_ns3/aligned/` with `__d2__` / `__d4__` in their names.

## 4. Tables and figures

```bash
python3 make_paper_results.py --results results_ns3 --latency ../python/results/latency_CPU.csv
```
