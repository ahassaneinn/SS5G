#!/usr/bin/env python3
"""
summarize_combo.py -- per model x scenario result files (Python 3 standard library only).

Safe to run at any time, also while simulations are still running: it only reads finished
run files (*.csv; files still being written are named *.csv.part and are ignored).

  python3 summarize_combo.py --results results_ns3 --scenario main --model GRU

Writes results_ns3/by_model/<scenario>/:
  <Model>_runs.csv      every finished run of this model in this scenario, one row per
                        run x group x direction, with model/seed/policy/cap/density columns
  <Model>_summary.csv   mean, sd, 95% CI and n per (density, policy, cap, group, dir)
  <Model>_summary.md    readable tables + allocation metrics + progress (done / expected runs)
"""
import argparse, csv, glob, json, math, os, re, time
from collections import defaultdict

# scenario name used by the user -> (internal scenario, densities)
SCEN = {"main": ("real", [1]), "aligned": ("aligned", [1]), "density": ("aligned", [2, 4])}
METRICS = ["goodput_mbps", "delivery_ratio", "loss_pct", "delay_mean_ms", "delay_p50_ms",
           "delay_p95_ms", "delay_p99_ms", "jitter_mean_ms", "share_delay_le_50ms",
           "share_delay_le_100ms", "stall_frac", "lag_mean_s", "flow_thr_mbps"]
ALLOC = ["pref_acc", "assign_acc", "lb_override_rate", "switch_rate", "n_servers_used",
         "overload_user_seconds_frac", "util_p95", "jain_server_bytes"]
TAG = re.compile(r"(?P<scenario>[a-z]+)__d(?P<density>\d+)__(?P<model>[A-Za-z]+)__s(?P<mseed>\d+)"
                 r"__(?P<policy>[a-z]+)__cap(?P<cap>[0-9.]+)(__run(?P<run>\d+))?$")
# two-sided 95% t critical values, df = 1..30 (1.96 beyond)
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def stats(vals):
    v = [x for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return None, None, None, 0
    m = sum(v) / n
    if n == 1:
        return m, None, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    t = T95[n - 2] if n - 1 <= 30 else 1.96
    return m, sd, t * sd / math.sqrt(n), n


def capname(c):
    return "unlimited" if float(c) == 0 else f"{float(c):g} Mbps"


def fmt(m, ci, d=2):
    if m is None:
        return "--"
    return f"{m:.{d}f}" if ci is None else f"{m:.{d}f} ± {ci:.{d}f}"


def expected_runs(results, scenario, model):
    """sum of planned runs from the job list written by run_ns3.sh (None if not found)"""
    tot, found = 0, False
    for f in glob.glob(os.path.join(results, f"jobs_{scenario}_*.txt")):
        if f.endswith(f"_{model}.txt"):
            found = True
            for line in open(f):
                p = line.split()
                if len(p) >= 7 and p[2] == model:
                    tot += int(p[6])
    return tot if found else None


def summarize(results, scenario, model, quiet=False):
    scen, dens_list = SCEN[scenario]
    files = []
    for f in glob.glob(os.path.join(results, scen, "net", f"{scen}__d*__{model}__s*__cap*__run*.csv")):
        m = TAG.match(os.path.basename(f)[:-4])
        if m and m.group("model") == model and int(m.group("density")) in dens_list:
            files.append((f, m))
    if not files and expected_runs(results, scenario, model) is None:
        return 0, None                      # not started yet: write nothing
    outdir = os.path.join(results, "by_model", scenario)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, model)

    # ---- merged runs file
    groups = defaultdict(lambda: defaultdict(list))   # (dens, policy, cap, group, dir) -> metric -> values
    header = None
    meta_cols = ["model", "scenario", "density", "mseed", "policy", "cap"]
    with open(base + "_runs.csv.tmp", "w", newline="") as out:
        w = None
        for f, m in sorted(files, key=lambda x: x[0]):
            try:
                rows = list(csv.DictReader(open(f)))
            except Exception:
                continue
            for r in rows:
                if header is None:
                    header = meta_cols + list(r.keys())
                    w = csv.DictWriter(out, fieldnames=header)
                    w.writeheader()
                r2 = {"model": model, "scenario": scenario, "density": m.group("density"),
                      "mseed": m.group("mseed"), "policy": m.group("policy"), "cap": m.group("cap")}
                r2.update(r)
                w.writerow({k: r2.get(k, "") for k in header})
                key = (int(m.group("density")), m.group("policy"), float(m.group("cap")), r["group"], r["dir"])
                for k in METRICS:
                    groups[key][k].append(fnum(r.get(k)))
    os.replace(base + "_runs.csv.tmp", base + "_runs.csv")

    # ---- summary csv
    order = lambda k: (k[0], k[1], k[2] != 0, -k[2], k[3], k[4])
    with open(base + "_summary.csv.tmp", "w", newline="") as out:
        w = csv.writer(out)
        hdr = ["model", "scenario", "density", "policy", "cap_mbps", "group", "dir", "n_runs"]
        for k in METRICS:
            hdr += [k, k + "_sd", k + "_ci95"]
        w.writerow(hdr)
        for key in sorted(groups, key=order):
            row = [model, scenario, key[0], key[1], key[2], key[3], key[4]]
            n_runs = 0
            cells = []
            for k in METRICS:
                m_, sd, ci, n = stats(groups[key][k])
                n_runs = max(n_runs, n)
                cells += ["" if m_ is None else f"{m_:.6g}", "" if sd is None else f"{sd:.6g}",
                          "" if ci is None else f"{ci:.6g}"]
            w.writerow(row + [n_runs] + cells)
    os.replace(base + "_summary.csv.tmp", base + "_summary.csv")

    # ---- allocation metrics
    alloc = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(results, scen, "alloc", f"{scen}__d*__{model}__s*__cap*.json")):
        m = TAG.match(os.path.basename(f)[:-5])
        if not m or m.group("model") != model or int(m.group("density")) not in dens_list:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        key = (int(m.group("density")), m.group("policy"), float(m.group("cap")))
        for k in ALLOC:
            alloc[key][k].append(fnum(d.get(k)))

    # ---- markdown
    n_done = len(files)
    exp = expected_runs(results, scenario, model)
    md = [f"# {model} — scenario `{scenario}`", "",
          f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')} — finished runs: **{n_done}**"
          + (f" of {exp} ({100 * n_done / exp:.0f} %)" if exp else ""), "",
          "Mean ± 95 % CI over runs (5 training seeds × ns-3 replications).", ""]
    cols = [("goodput_mbps", "Goodput (Mbps)", 2), ("delivery_ratio", "Delivered", 3),
            ("delay_mean_ms", "Delay mean (ms)", 1), ("delay_p95_ms", "Delay p95 (ms)", 1),
            ("delay_p99_ms", "Delay p99 (ms)", 1), ("jitter_mean_ms", "Jitter (ms)", 2),
            ("loss_pct", "Loss (%)", 2)]

    def table(title, sel, cols_):
        keys = [k for k in sorted(groups, key=order) if sel(k)]
        if not keys:
            return
        md.extend([f"## {title}", "", "| density | policy | capacity | runs | " + " | ".join(c[1] for c in cols_) + " |",
                   "|---|---|---|---|" + "---|" * len(cols_)])
        for k in keys:
            vals = []
            n_runs = 0
            for c, _, d in cols_:
                m_, sd, ci, n = stats(groups[k][c])
                n_runs = max(n_runs, n)
                vals.append(fmt(m_, ci, d))
            md.append(f"| x{k[0]} | {k[1]} | {capname(k[2])} | {n_runs} | " + " | ".join(vals) + " |")
        md.append("")

    table("All traffic", lambda k: k[3] == "ALL" and k[4] == "ALL", cols)
    table("Video downlink", lambda k: k[3] == "VIDEO" and k[4] == "DL",
          [("goodput_mbps", "Goodput (Mbps)", 2), ("delivery_ratio", "Delivered", 3),
           ("stall_frac", "Stall share", 3), ("lag_mean_s", "Lag (s)", 2), ("delay_p95_ms", "Delay p95 (ms)", 1)])
    table("Gaming (all directions)", lambda k: k[3] == "GAMING" and k[4] == "ALL",
          [("delay_mean_ms", "Delay mean (ms)", 1), ("delay_p95_ms", "Delay p95 (ms)", 1),
           ("jitter_mean_ms", "Jitter (ms)", 2), ("loss_pct", "Loss (%)", 2),
           ("share_delay_le_50ms", "Pkts ≤ 50 ms", 3)])
    if alloc:
        md.extend(["## Allocation metrics (mean over seeds)", "",
                   "| density | policy | capacity | acc. top-1 | acc. after LB | LB override | switch rate | "
                   "servers | overloaded user-s | util p95 | Jain |", "|---|---|---|---|---|---|---|---|---|---|---|"])
        for k in sorted(alloc, key=lambda k: (k[0], k[1], k[2] != 0, -k[2])):
            a = {c: stats(alloc[k][c])[0] for c in ALLOC}
            f3 = lambda v, d=3: "--" if v is None else f"{v:.{d}f}"
            md.append(f"| x{k[0]} | {k[1]} | {capname(k[2])} | {f3(a['pref_acc'])} | {f3(a['assign_acc'])} | "
                      f"{f3(a['lb_override_rate'])} | {f3(a['switch_rate'])} | {f3(a['n_servers_used'], 0)} | "
                      f"{f3(a['overload_user_seconds_frac'])} | {f3(a['util_p95'], 2)} | {f3(a['jain_server_bytes'])} |")
        md.append("")
    open(base + "_summary.md", "w").write("\n".join(md) + "\n")
    if not quiet:
        print(f"[summary] {scenario}/{model}: {n_done}" + (f"/{exp}" if exp else "") + f" runs -> {base}_summary.md")
    return n_done, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_ns3")
    ap.add_argument("--scenario", choices=list(SCEN), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    summarize(a.results, a.scenario, a.model, a.quiet)


if __name__ == "__main__":
    main()
