#!/usr/bin/env python3
r"""
analyze_ns3.py -- ns-3 stage 4: aggregate all runs into tables and figures.

  python3 analyze_ns3.py --results results_ns3 [--latency-cpu ../python/results/latency_CPU.csv]

Every number is reported as mean +- sd and a 95% t-confidence interval over runs
(DL models: 5 training seeds x RUNS_DL ns-3 replications; baselines: RUNS_BASE replications).
Welch t-tests (Holm-corrected within each scenario/cap/policy) compare every allocator
with the persistence baseline and with the observed (historical) assignment.

Outputs in <results>/report/:
  summary_ns3.csv          one row per scenario/density/model/policy/cap/group/dir (all metrics)
  alloc_summary.csv        allocation-level metrics (override, switch rate, overload, imbalance)
  tests_ns3.csv            Welch t-tests
  REPORT_ns3.md            readable tables
  table_<scenario>_*.tex   LaTeX tables (booktabs)
  fig_<scenario>_*.pdf/png figures with 95% CI error bars
Needs: pandas, numpy, scipy, matplotlib.
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
from scipy import stats

ORDER = ["Observed", "Persistence", "UserMajority", "AppMajority", "NearestServer", "LBOnly",
         "DNN", "LSTM", "GRU", "Transformer", "Autoformer", "TFT", "FEDformer",
         "DeepCog", "Dong", "DeepSlice", "Chen"]
LABEL = {"Observed": "Observed (historical)", "LBOnly": "LB only (no prediction)",
         "NearestServer": "Nearest server", "UserMajority": "Per-user majority",
         "AppMajority": "Per-app majority"}
METRICS = ["goodput_mbps", "delivery_ratio", "flow_thr_mbps", "loss_pct", "delay_mean_ms",
           "delay_p50_ms", "delay_p95_ms", "delay_p99_ms", "jitter_mean_ms",
           "share_delay_le_50ms", "share_delay_le_100ms", "stall_frac", "lag_mean_s"]
ALLOC = ["pref_acc", "assign_acc", "lb_override_rate", "switch_rate", "n_servers_used",
         "overload_user_seconds_frac", "overload_bytes_frac", "util_p95",
         "load_imbalance_max_over_mean", "jain_server_bytes"]
TAG = re.compile(r"(?P<smoke>smoke_)?(?P<scenario>[a-z]+)__d(?P<density>\d+)__(?P<model>[A-Za-z]+)__s(?P<mseed>\d+)"
                 r"__(?P<policy>[a-z]+)__cap(?P<cap>[0-9.]+)(__run(?P<run>\d+))?")


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = x.mean()
    if n == 1:
        return m, np.nan, np.nan, 1
    sd = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, h, n


def capname(c):
    c = float(c)
    return "unlimited" if c == 0 else f"{c:g} Mbps"


def load(results, include_smoke):
    rows = []
    for f in glob.glob(os.path.join(results, "*", "net", "*.csv")):
        m = TAG.match(os.path.basename(f)[:-4])
        if not m or (m.group("smoke") and not include_smoke):
            continue
        d = pd.read_csv(f)
        for k in ["scenario", "density", "model", "mseed", "policy", "cap"]:
            d[k] = m.group(k)
        d["run"] = int(m.group("run"))
        rows.append(d)
    if not rows:
        raise SystemExit(f"no result files under {results}/*/net/")
    net = pd.concat(rows, ignore_index=True)
    net["density"] = net.density.astype(int)
    net["cap"] = net.cap.astype(float)
    al = []
    for f in glob.glob(os.path.join(results, "*", "alloc", "*.json")):
        m = TAG.match(os.path.basename(f)[:-5])
        if not m or (m.group("smoke") and not include_smoke):
            continue
        d = json.load(open(f))
        d.update({k: m.group(k) for k in ["scenario", "model", "mseed", "policy"]})
        d["density"] = int(m.group("density"))
        d["cap"] = float(m.group("cap"))
        al.append(d)
    return net, pd.DataFrame(al)


def summarise(net, latency):
    keys = ["scenario", "density", "model", "policy", "cap", "group", "dir"]
    out = []
    for k, g in net.groupby(keys):
        r = dict(zip(keys, k))
        r["n_runs"] = len(g)
        r["n_model_seeds"] = g.mseed.nunique()
        for mtr in METRICS:
            m, sd, h, n = ci95(g[mtr])
            r[mtr], r[mtr + "_sd"], r[mtr + "_ci"] = m, sd, h
        inf = latency.get(r["model"], 0.0)
        r["inference_ms"] = inf
        # total latency = mean one-way network delay + per-decision inference latency (batch 1)
        tot = g.delay_mean_ms + inf
        m, sd, h, n = ci95(tot)
        r["total_latency_ms"], r["total_latency_ms_sd"], r["total_latency_ms_ci"] = m, sd, h
        out.append(r)
    s = pd.DataFrame(out)
    s["_o"] = s.model.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return s.sort_values(["scenario", "density", "cap", "policy", "group", "dir", "_o"]).drop(columns="_o")


def tests(net):
    out = []
    keys = ["scenario", "density", "policy", "cap"]
    sub = net[(net.group == "ALL") & (net.dir == "ALL")]
    for k, g in sub.groupby(keys):
        for ref in ["Persistence", "Observed"]:
            gr = g[g.model == ref]
            if gr.empty:
                continue
            for mtr in ["delay_mean_ms", "delay_p95_ms", "goodput_mbps", "delivery_ratio", "loss_pct"]:
                res = []
                for model, gm in g.groupby("model"):
                    if model == ref:
                        continue
                    a, b = gm[mtr].dropna(), gr[mtr].dropna()
                    if len(a) < 2 or len(b) < 2:
                        continue
                    t, p = stats.ttest_ind(a, b, equal_var=False)
                    res.append([model, a.mean() - b.mean(), p])
                if not res:
                    continue
                # Holm correction
                ps = np.array([x[2] for x in res])
                order = np.argsort(ps)
                adj = np.empty_like(ps)
                run = 0
                for i, j in enumerate(order):
                    run = max(run, min(1, (len(ps) - i) * ps[j]))
                    adj[j] = run
                for (model, diff, p), pa in zip(res, adj):
                    out.append(dict(zip(keys, k), reference=ref, metric=mtr, model=model,
                                    diff_mean=diff, p_welch=p, p_holm=pa, significant=pa < 0.05))
    return pd.DataFrame(out)


def fmt(m, ci, d=2):
    if pd.isna(m):
        return "--"
    return f"{m:.{d}f}" if pd.isna(ci) else f"{m:.{d}f} ± {ci:.{d}f}"


def tables(s, al, outdir):
    md = ["# ns-3 results (mean ± 95% CI over runs)\n"]
    cols = [("goodput_mbps", "Goodput (Mbps)", 2), ("delivery_ratio", "Delivered", 3),
            ("delay_mean_ms", "Delay mean (ms)", 1), ("delay_p95_ms", "Delay p95 (ms)", 1),
            ("delay_p99_ms", "Delay p99 (ms)", 1), ("jitter_mean_ms", "Jitter (ms)", 2),
            ("loss_pct", "Loss (%)", 2), ("total_latency_ms", "Total latency (ms)", 1)]
    for (scen, dens), gs in s.groupby(["scenario", "density"]):
        for pol in ["direct", "lb", "lbk"]:
            for cap in sorted(gs.cap.unique(), reverse=True, key=lambda c: (c == 0, c)):
                base = gs[(gs.cap == cap) & (gs.group == "ALL") & (gs.dir == "ALL")]
                g = base[base.policy == pol]
                if pol == "lb":
                    g = pd.concat([g, base[base.policy == "leastload"]])   # LB-only shown next to LB
                if g.empty:
                    continue
                title = f"{scen} (density x{dens}) -- policy {pol} -- capacity {capname(cap)}"
                md.append(f"\n## {title}\n")
                md.append("| Allocator | runs | " + " | ".join(c[1] for c in cols) + " |")
                md.append("|---|---|" + "---|" * len(cols))
                tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
                       "Allocator & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
                for _, r in g.iterrows():
                    name = LABEL.get(r.model, r.model)
                    vals = [fmt(r[c], r.get(c + "_ci"), d) for c, _, d in cols]
                    md.append(f"| {name} | {r.n_runs} | " + " | ".join(vals) + " |")
                    tex.append(name + " & " + " & ".join(v.replace("±", "$\\pm$") for v in vals) + " \\\\")
                tex += ["\\bottomrule", "\\end{tabular}"]
                fn = f"table_{scen}_d{dens}_{pol}_cap{cap:g}.tex"
                open(os.path.join(outdir, fn), "w").write("\n".join(tex))
        # per-application QoS: video stall proxy, gaming delay/jitter/loss
        for cap in sorted(gs.cap.unique(), reverse=True, key=lambda c: (c == 0, c)):
            v = gs[(gs.policy.isin(["direct", "lb"])) & (gs.cap == cap) & (gs.group == "VIDEO") & (gs.dir == "DL")]
            gm = gs[(gs.policy.isin(["direct", "lb"])) & (gs.cap == cap) & (gs.group == "GAMING") & (gs.dir == "ALL")]
            if v.empty and gm.empty:
                continue
            md.append(f"\n## {scen} (x{dens}) -- per-application QoS -- capacity {capname(cap)}\n")
            md.append("| Allocator | policy | Video goodput (Mbps) | Video stall share | Video lag (s) | "
                      "Gaming delay p95 (ms) | Gaming jitter (ms) | Gaming loss (%) | Gaming pkts <= 50 ms |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for (model, pol), _ in v.groupby(["model", "policy"], sort=False):
                rv = v[(v.model == model) & (v.policy == pol)].iloc[0]
                rg = gm[(gm.model == model) & (gm.policy == pol)]
                rg = rg.iloc[0] if len(rg) else None
                g_ = (lambda c, d: fmt(rg[c], rg[c + "_ci"], d)) if rg is not None else (lambda c, d: "--")
                md.append(f"| {LABEL.get(model, model)} | {pol} | {fmt(rv.goodput_mbps, rv.goodput_mbps_ci)} | "
                          f"{fmt(rv.stall_frac, rv.stall_frac_ci, 3)} | {fmt(rv.lag_mean_s, rv.lag_mean_s_ci)} | "
                          f"{g_('delay_p95_ms', 1)} | {g_('jitter_mean_ms', 2)} | {g_('loss_pct', 2)} | "
                          f"{g_('share_delay_le_50ms', 3)} |")
    if not al.empty:
        md.append("\n## Allocation-level metrics (mean over model seeds)\n")
        a = al.groupby(["scenario", "density", "cap", "policy", "model"])[ALLOC].mean().reset_index()
        md.append("| scenario | x | cap | policy | allocator | acc. top-1 | acc. after LB | LB override | "
                  "switch rate | servers | overloaded user-s | util p95 | Jain |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in a.iterrows():
            md.append(f"| {r.scenario} | {r.density} | {capname(r.cap)} | {r.policy} | {LABEL.get(r.model, r.model)} | "
                      f"{r.pref_acc:.3f} | {r.assign_acc:.3f} | {r.lb_override_rate:.3f} | {r.switch_rate:.3f} | "
                      f"{r.n_servers_used:.0f} | {r.overload_user_seconds_frac:.3f} | {r.util_p95:.2f} | "
                      f"{r.jain_server_bytes:.3f} |")
        a.to_csv(os.path.join(outdir, "alloc_summary.csv"), index=False)
    open(os.path.join(outdir, "REPORT_ns3.md"), "w").write("\n".join(md) + "\n")


def figures(s, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
                         "legend.fontsize": 11, "lines.linewidth": 2.0, "lines.markersize": 7})
    marks = "osD^v<>phX*P+"
    for (scen, dens), gs in s.groupby(["scenario", "density"]):
        caps = sorted(gs.cap.unique(), key=lambda c: (c == 0, c))
        xt = [capname(c) for c in caps]
        for pol in ["direct", "lb"]:
            for mtr, lab in [("delay_mean_ms", "Mean one-way delay (ms)"), ("delay_p95_ms", "95th-pct delay (ms)"),
                             ("goodput_mbps", "Goodput per active user (Mbps)"),
                             ("total_latency_ms", "Total latency = delay + inference (ms)"),
                             ("loss_pct", "Packet loss (%)")]:
                fig, ax = plt.subplots(figsize=(9, 5.5))
                g = gs[(gs.group == "ALL") & (gs.dir == "ALL") &
                       ((gs.policy == pol) | ((gs.cap == 0) & (gs.policy == "direct")) |
                        ((pol == "lb") & (gs.policy == "leastload")))]
                if g.empty:
                    plt.close(fig)
                    continue
                for i, model in enumerate([m for m in ORDER if m in set(g.model)]):
                    gm = g[g.model == model].set_index("cap").reindex(caps)
                    ax.errorbar(range(len(caps)), gm[mtr], yerr=gm[mtr + "_ci"], marker=marks[i % len(marks)],
                                capsize=4, label=LABEL.get(model, model))
                ax.set_xticks(range(len(caps)))
                ax.set_xticklabels(xt)
                ax.set_xlabel("Per-server capacity")
                ax.set_ylabel(lab)
                ax.grid(alpha=0.3)
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                fig.tight_layout()
                base = os.path.join(outdir, f"fig_{scen}_d{dens}_{pol}_{mtr}")
                fig.savefig(base + ".pdf")
                fig.savefig(base + ".png", dpi=150)
                plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_ns3")
    ap.add_argument("--latency-cpu", default=None, help="latency_CPU.csv from eua_pipeline.py (batch-1 median used)")
    ap.add_argument("--include-smoke", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.results, "report")
    os.makedirs(out, exist_ok=True)
    lat = {}
    if a.latency_cpu and os.path.exists(a.latency_cpu):
        L = pd.read_csv(a.latency_cpu)
        lat = dict(zip(L.model, L.latency_b1_median_ms))
    net, al = load(a.results, a.include_smoke)
    s = summarise(net, lat)
    s.to_csv(os.path.join(out, "summary_ns3.csv"), index=False)
    t = tests(net)
    t.to_csv(os.path.join(out, "tests_ns3.csv"), index=False)
    tables(s, al, out)
    figures(s, out)
    print(f"{len(net)} result rows from {net.groupby(['scenario','model','mseed','policy','cap','run']).ngroups} runs"
          f" -> {out}")


if __name__ == "__main__":
    main()
