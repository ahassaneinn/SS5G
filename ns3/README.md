# Trace-driven ns-3 evaluation

These scripts replay the server assignments predicted in `../python` in an edge-server-level, trace-driven ns-3 simulation. Every allocator gets exactly the same traffic.

| File | What it does |
|---|---|
| `make_ns3_inputs.py` | Builds `ns3_inputs/` (test-period traffic, servers, users, ranked decisions of every model and baseline) from `../python/work` and `../python/results`. `run_all` in `../python` calls it as its last step. |
| `allocate.py` | Allocation and load-balancing rule (standard library only). Turns one model's decisions into a flow schedule. |
| `eua_trace_sim.cc` | ns-3 program that replays a flow schedule and measures QoS |
| `run_all_models.sh` | One command for all models, reference allocators and scenarios |
| `run_ns3.sh` | Runs one preset (allocators × capacities × policies × replications) in parallel |
| `summarize_combo.py` | Per model/scenario summaries (standard library only; safe to run while simulations are running) |
| `analyze_ns3.py` | Mean ± sd and 95% CI tables, Welch t-tests (Holm), and figures with error bars |

## 1. Set up (once)

You need ns-3.36 or newer (CMake-based `./ns3` script) and Python 3. `analyze_ns3.py` also needs pandas, scipy and matplotlib; it can run on another machine.

```bash
# copy this folder (with ns3_inputs/) to the Linux machine, e.g. ~/eua_ns3/
cd ~/eua_ns3 && chmod +x *.sh
# on "bad interpreter" or "\r" errors (Windows line endings):  sed -i 's/\r$//' *.sh *.py

cp eua_trace_sim.cc ~/ns-3-dev/scratch/          # your ns-3 folder
cd ~/ns-3-dev
./ns3 configure --build-profile=optimized --disable-examples --disable-tests   # recommended, much faster
./ns3 build scratch/eua_trace_sim
```

## 2. Smoke test (a few minutes)

```bash
cd ~/eua_ns3
NS3_DIR=~/ns-3-dev ./run_ns3.sh smoke
```

The smoke test uses 7 users (one per application), the Observed and GRU allocators, unlimited capacity and 10 Mbps, and `direct` and `lb`, with 1 run each. Every run prints a line like `DONE run=1 delivery=0.99 delay_mean_ms=... wall_s=...`.

To run one simulation by hand:

```bash
python3 allocate.py --decisions GRU__s0 --policy lb --cap 10 --scenario real --out /tmp/f.csv --stats /tmp/a.json
cd ~/ns-3-dev && ./ns3 run "eua_trace_sim --flows=/tmp/f.csv --cap=10 --run=1 --out=/tmp/r.csv"
```

## 3. Full evaluation

```bash
cd ~/eua_ns3
NS3_DIR=~/ns-3-dev JOBS=10 nohup ./run_all_models.sh > run_all.out 2>&1 &   # keeps running after logout
./run_all_models.sh status                                                  # progress, at any time
tail -f results_ns3/run_all.log
```

The script runs scenario by scenario (`main` → `aligned` → `density`). Within each scenario it runs the 11 DL models, then the reference allocators (Observed, Persistence, LBOnly, UserMajority, AppMajority, NearestServer). Each model × scenario combination writes its own files to `results_ns3/by_model/<scenario>/`:

- `<Model>_summary.md`: readable tables, refreshed every `REFRESH` seconds while the combination runs;
- `<Model>_summary.csv`: mean, sd and 95% CI per density × policy × capacity × group × direction;
- `<Model>_runs.csv`: every finished run.

You can stop the script with Ctrl+C at any time. Running the same command again resumes it and skips finished runs. At the end, it runs `analyze_ns3.py` if pandas, scipy and matplotlib are installed:

```bash
python3 analyze_ns3.py --results results_ns3 --latency-cpu ../python/results/latency_CPU.csv
```

`analyze_ns3.py` writes `results_ns3/report/`, which contains `REPORT_ns3.md`, CSV and LaTeX tables, tests and figures.

### Options (environment variables)

| Variable | Meaning | Default |
|---|---|---|
| `NS3_DIR` | ns-3 tree with `eua_trace_sim` built | `~/ns-3-dev` |
| `JOBS` | parallel simulations | cores − 1 |
| `SCENARIOS` | any of `main aligned density` | all three |
| `MODELS` | DL models to run | all 11 |
| `REFERENCE` | reference allocators (`""` for none) | all six |
| `RUNS_DL` | ns-3 replications per training seed (5 seeds) | 4 (20 runs per model) |
| `RUNS_BASE` | replications per deterministic baseline | 20 |
| `CAPS` | override the capacity list, e.g. `"0 10"` | per preset |
| `EXTRA` | extra flags for `eua_trace_sim`, e.g. `"--qdisc=codel"` | none |

To run the presets separately, use `./run_ns3.sh main | aligned | density`. `./run_ns3.sh list` prints the number of runs in each preset.

## 4. What is simulated

**Scenarios.**

| Preset | Description |
|---|---|
| `main` | Every user-second is replayed at its recorded time. Per-server capacities: unlimited, 40, 10, 5 and 1 Mbps. |
| `aligned` | All users' test periods start at t = 0 (up to 83 concurrent users). Capacities: unlimited, 100, 50 and 25 Mbps. |
| `density` | The aligned scenario with every user replicated ×2 and ×4, at 50 Mbps. |

**Traffic.** For each test-period user-second, the simulator replays the bytes that the user exchanged with its serving server in that second, at that second. Every allocator gets exactly the same bytes.

- Video traffic over QUIC is replayed over TCP (`--video-udp tcp`), because QUIC is congestion-controlled.
- Gaming traffic stays UDP, with 200-byte payloads.
- TCP bytes are written to the socket in pieces at random offsets within the second, and unsent bytes carry over to the next second.
- UDP packets are paced with random gaps.

**Decisions.** For each user-second, the allocator chooses a server:

- `Observed`: the historical server;
- `Persistence`: the current server;
- `UserMajority`, `AppMajority` and `NearestServer`: the non-learning baselines;
- the DL models: their top-1 server, or their top-5 candidates with `lbk`;
- `LBOnly`: no prediction.

**Allocation policies** (`allocate.py`; the header documents the full rule):

| Policy | Rule |
|---|---|
| `direct` | The model's top-1 server, without a capacity check. |
| `lb` | The top-1 server if it has spare capacity. Otherwise, the first server with spare capacity among the user's previous server (if it belongs to the same provider) and the 10 nearest servers of the same provider. If none has spare capacity, the least-loaded of these servers, with ties resolved in favor of the model's choice. |
| `lbk` | Like `lb`, but tries the model's candidates 2–5 before the provider list. |
| `leastload` | No prediction: the nearest server of the user's current provider with spare capacity, otherwise the least-loaded one. |

Demand is estimated causally from the user's previous active second, and users are processed in ascending ID order in 1 s epochs. A user-second is never split. A reassigned user opens a new connection while the old one drains.

**Topology and parameters.** The topology is a star: user — access link — gateway — server link — server.

| Parameter | Value (CLI flag) |
|---|---|
| Access link | 1 Gbps; one-way delay U(5, 10) ms per user and run (`--accessMbps --accessMinMs --accessMaxMs`) |
| Server link | `--cap` Mbps in both directions (10 Gbps when unlimited) |
| Propagation delay | great-circle distance × 1.5 / (200 km/ms), at least 0.5 ms (`allocate.py --stretch --fiber-km-per-ms`) |
| Queues | FIFO drop-tail, 100 ms at link rate, 10–10,000 packets (`--qdisc --bufferMs --minQ --maxQ`) |
| TCP | CUBIC, MSS 1448 B, 4 MB buffers (`--tcp --tcpBuf`) |
| UDP payload | 1200 B; 200 B for gaming (`--udpPayload --gamingPayload`) |
| Replications | `--run` = 1…N (RngRun), the same run numbers for every allocator |
| Drain | 30 s after the last second (`--drain`) |

**Metrics.** The network metrics are computed per run for all traffic, video, gaming and each application, and for downlink, uplink and both:

- delivered vs. offered bytes and goodput;
- per-flow throughput;
- packet loss;
- one-way delay (mean, p50, p95, p99) and jitter;
- the share of packets delivered within 50 and 100 ms;
- a video stall proxy: the share of seconds not fully delivered within 2 s.

The allocation metrics are:

- accuracy before and after LB;
- LB override rate and server switch rate;
- servers used;
- overloaded user-seconds and bytes;
- p95 utilization, load imbalance, and Jain's index.

**Scope.** This is an edge-server-level simulation. The radio access network is represented by a per-user access delay. The simulation does not model the radio stack, fading, interference, scheduling, handover or mobility (users are stationary). Server capacity is an egress rate.
