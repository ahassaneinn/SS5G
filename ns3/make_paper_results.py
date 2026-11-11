#!/usr/bin/env python3
"""make_paper_results.py -- tables and figures of the ns-3 evaluation (JCC revision 2, SS5G v3.0).

Run in this folder after the simulations (run_everything.sh) have finished:

    python3 make_paper_results.py [--results results_ns3] [--latency ../python/results/latency_CPU.csv]

Writes to paper_results/:
    tab08_trace.tex        Table 8  mean delay (direct/LB) and delivered traffic at 1 Mbps, trace timing
    tab09_contention.tex   Table 9  mean delay under contention (aligned, x2, x4)
    tab10_goodput.tex      Table 10 goodput under contention
    tab11_paired.tex       Table 11 paired differences between allocators (mean +- 95% CI over paired runs)
    tab12_residual.tex     Table 12 GRU with LB vs residual-aware LB (LB-R)
    tab13_loadvar.tex      Table 13 utilization variance and reassignment rate
    variance_GRU.csv       training vs network variance components (GRU, 3 x 3 seeds, two-way ANOVA)
    ci_max.txt             largest 95% CI half-widths (table captions)
    Fig*.pdf / *.png       Figures 1-6 (goodput, delay, total latency; overview and zoom)
Needs pandas, numpy, scipy and matplotlib.
Learned models: run r uses training seed r-1 (the 'diagonal' runs); GRU also has every seed x run.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats

PAT = re.compile(r"(?P<scen>[a-z]+)__d(?P<dens>\d+)__(?P<model>[A-Za-z]+)__(?P<seed>s\d+)__(?P<pol>[a-z]+)__cap(?P<cap>[\d.]+)(__run(?P<run>\d+))?\.(csv|json)$")
BASE = ["Observed", "Persistence", "UserMajority", "AppMajority", "NearestServer"]
DL = ["DNN", "LSTM", "GRU", "Transformer", "Autoformer", "TFT", "FEDformer"]
LIT = ["DeepCog", "Dong", "DeepSlice", "Chen"]
LAB = {"Observed": "Observed", "Persistence": "Persistence", "UserMajority": "User majority",
       "AppMajority": "App majority", "NearestServer": "Nearest server", "LBOnly": "LB only"}
SETS = [("real", 1, 40, "T-40"), ("real", 1, 1, "T-1"), ("aligned", 1, 100, "A-100"), ("aligned", 1, 50, "A-50"),
        ("aligned", 1, 25, "A-25"), ("aligned", 2, 50, "$\\times$2"), ("aligned", 4, 50, "$\\times$4")]


def tag(f):
    m = PAT.match(os.path.basename(f))
    if not m:
        return None
    g = m.groupdict()
    return dict(scen=g["scen"], dens=int(g["dens"]), model=g["model"], seed=g["seed"], sn=int(g["seed"][1:]),
                pol=g["pol"], cap=float(g["cap"]), run=int(g["run"]) if g["run"] else None)


def load_csv(res, sub, group=None):
    rows = []
    for f in glob.glob(os.path.join(res, "*", sub, "*.csv")):
        t = tag(f)
        if not t:
            continue                      # smoke runs and others
        d = pd.read_csv(f)
        if group is not None:
            d = d[(d.group == group) & (d.dir == "ALL")]
        for k, v in t.items():
            d[k] = v
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_alloc(res):
    rows = []
    for f in glob.glob(os.path.join(res, "*", "alloc", "*.json")):
        t = tag(f)
        if t:
            j = json.load(open(f)); j.update(t); rows.append(j)
    return pd.DataFrame(rows)


def diag(d):
    """reference allocators: all runs; learned models: run r with training seed r-1"""
    return d[(~d.model.isin(DL + LIT)) | (d.run == d.sn + 1)]


def ci(x):
    x = np.asarray(x, float); n = len(x)
    return stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan


def sel(d, scen, dens, model, pol, cap):
    return d[(d.scen == scen) & (d.dens == dens) & (d.model == model) & (d.pol == pol) & (d.cap == cap)]


def mean(d, scen, dens, model, pol, cap, met, scale=1):
    x = sel(d, scen, dens, model, pol, cap)[met]
    return x.mean() * scale if len(x) else np.nan


def fmt(v, f="{:.1f}"):
    return "--" if np.isnan(v) else f.format(v)


def paired(d, scen, dens, A, B, cap, met, scale=1):
    a = sel(d, scen, dens, A[0], A[1], cap).set_index("run")[met] * scale
    b = sel(d, scen, dens, B[0], B[1], cap).set_index("run")[met] * scale
    x = (a - b).dropna()
    return (x.mean(), ci(x)) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_ns3")
    ap.add_argument("--latency", default="../python/results/latency_CPU.csv")
    ap.add_argument("--out", default="paper_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net_all = load_csv(a.results, "net", "ALL")
    net = diag(net_all)
    srv = diag(load_csv(a.results, "srv"))
    alc = load_alloc(a.results)
    alc = alc[(~alc.model.isin(DL + LIT)) | alc.seed.isin(["s0", "s1", "s2"])]
    W = lambda name, lines: open(os.path.join(a.out, name), "w").write("\n".join(lines) + "\n")
    notes = []

    # ---- Table 8
    out = []
    for grp in (BASE, DL, LIT):
        for m in grp:
            c = [mean(net, "real", 1, m, "direct", cap, "delay_mean_ms") for cap in [0, 40, 10, 5, 1]]
            l = [mean(net, "real", 1, m, "lb", cap, "delay_mean_ms") for cap in [40, 10, 5, 1]]
            dd = mean(net, "real", 1, m, "direct", 1, "delivery_ratio", 100); dl = mean(net, "real", 1, m, "lb", 1, "delivery_ratio", 100)
            out.append(LAB.get(m, m) + " & " + " & ".join(fmt(v) for v in c + l) + f" & {fmt(dd)} / {fmt(dl)} \\\\")
        out.append("\\midrule")
    l = [mean(net, "real", 1, "LBOnly", "leastload", cap, "delay_mean_ms") for cap in [40, 10, 5, 1]]
    out.append("LB only & -- & -- & -- & -- & -- & " + " & ".join(fmt(v) for v in l)
               + f" & -- / {fmt(mean(net, 'real', 1, 'LBOnly', 'leastload', 1, 'delivery_ratio', 100))} \\\\")
    W("tab08_trace.tex", out)
    g = net[(net.scen == "real") & net.pol.isin(["direct", "lb", "leastload"])].groupby(["model", "pol", "cap"])
    notes.append(f"Table 8: max 95% CI delay {g.delay_mean_ms.apply(ci).max():.2f} ms, delivered {(g.delivery_ratio.apply(ci) * 100).max():.2f} pp")

    # ---- Tables 9 and 10
    cs = [("aligned", 1, 100), ("aligned", 1, 50), ("aligned", 1, 25), ("aligned", 2, 50), ("aligned", 4, 50)]
    for met, fn, f in [("delay_mean_ms", "tab09_contention.tex", "{:.1f}"), ("goodput_mbps", "tab10_goodput.tex", "{:.2f}")]:
        out = []
        for grp in (BASE, DL, LIT):
            for m in grp:
                v = []
                for s, dn, cap in cs:
                    v += [mean(net, s, dn, m, "direct", cap, met), mean(net, s, dn, m, "lb", cap, met)]
                out.append(LAB.get(m, m) + " & " + " & ".join(fmt(x, f) for x in v) + " \\\\")
            out.append("\\midrule")
        v = []
        for s, dn, cap in cs:
            v += ["--", fmt(mean(net, s, dn, "LBOnly", "leastload", cap, met), f)]
        out.append("LB only & " + " & ".join(v) + " \\\\")
        W(fn, out)
    C = net[(net.scen == "aligned") & net.pol.isin(["direct", "lb", "leastload"])]
    gc = C.groupby(["dens", "model", "pol", "cap"])
    notes.append(f"Tables 9/10: max 95% CI delay {gc.delay_mean_ms.apply(ci).max():.2f} ms, goodput {gc.goodput_mbps.apply(ci).max():.3f} Mbps")

    # ---- variance components (GRU, every training seed x every network seed)
    rows = []
    for (scen, dens, pol, cap), x in net_all[(net_all.model == "GRU") & (net_all.dens == 1) & net_all.pol.isin(["direct", "lb"])].groupby(["scen", "dens", "pol", "cap"]):
        for met, sc in [("delay_mean_ms", 1), ("delivery_ratio", 100)]:
            t = x.pivot_table(index="sn", columns="run", values=met)
            if t.shape != (3, 3):
                continue
            Y = t.values * sc; r_, c_ = Y.shape; gm = Y.mean()
            msT = c_ * ((Y.mean(1) - gm) ** 2).sum() / (r_ - 1); msN = r_ * ((Y.mean(0) - gm) ** 2).sum() / (c_ - 1)
            res = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm; msE = (res ** 2).sum() / ((r_ - 1) * (c_ - 1))
            vT = max(0, (msT - msE) / c_); vN = max(0, (msN - msE) / r_)
            rows.append(dict(scenario=scen, policy=pol, cap=cap, metric=met, mean=gm, sd_training=np.sqrt(vT),
                             sd_network=np.sqrt(vN), sd_residual=np.sqrt(msE)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "variance_GRU.csv"), index=False)

    # ---- Table 11 paired differences
    comps = [("GRU $-$ Observed (D)", ("GRU", "direct"), ("Observed", "direct")),
             ("GRU+LB $-$ GRU (D)", ("GRU", "lb"), ("GRU", "direct")),
             ("GRU+LB $-$ LB only", ("GRU", "lb"), ("LBOnly", "leastload")),
             ("GRU+LB $-$ Pers.+LB", ("GRU", "lb"), ("Persistence", "lb")),
             ("GRU+LB-R $-$ Pers.+LB-R", ("GRU", "lbres"), ("Persistence", "lbres"))]
    def cell(m, h):
        if np.isnan(m):
            return "--"
        if abs(m) < 1e-9 and (np.isnan(h) or abs(h) < 1e-9):
            return "0"
        return f"{m:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0").replace("-", "$-$") + f"$\\pm${h:.1f}"
    out = []
    for met, title, sc in [("delay_mean_ms", "Mean one-way delay (ms)", 1), ("delivery_ratio", "Delivered traffic (percentage points)", 100)]:
        out.append("\\multicolumn{8}{l}{\\textit{" + title + "}} \\\\")
        for lab, A, B in comps:
            out.append(lab + " & " + " & ".join(cell(*paired(net, s, dn, A, B, cap, met, sc)) for s, dn, cap, _ in SETS) + " \\\\")
        out.append("\\midrule")
    W("tab11_paired.tex", out[:-1])

    # ---- Table 12 residual traffic (GRU, LB vs LB-R)
    out = []
    for s, dn, cap, lab in [("real", 1, c, f"Trace, {c}") for c in (40, 10, 5, 1)] + [x for x in SETS if x[0] == "aligned"]:
        cells = []
        for met, d_, sc, f in [("false_admission_rate", srv, 100, "{:.1f}"), ("resid_mean_epochs", srv, 1, None),
                               ("delivery_ratio", net, 100, "{:.1f}"), ("delay_mean_ms", net, 1, "{:.1f}"), ("delay_p95_ms", net, 1, "{:.0f}")]:
            for pol in ("lb", "lbres"):
                v = mean(d_, s, dn, "GRU", pol, cap, met, sc)
                ff = f or ("{:,.0f}" if v >= 100 else "{:.1f}" if v >= 10 else "{:.2f}")
                cells.append(fmt(v, ff))
        out.append(lab + " & " + " & ".join(cells) + " \\\\")
    W("tab12_residual.tex", out)

    # ---- Table 13 utilization variance and reassignment rate
    rws = [("Observed (D)", "Observed", "direct"), ("Observed + LB", "Observed", "lb"), ("Observed + LB-R", "Observed", "lbres"),
           ("Persistence (D)", "Persistence", "direct"), ("Persistence + LB", "Persistence", "lb"), ("Persistence + LB-R", "Persistence", "lbres"),
           ("GRU (D)", "GRU", "direct"), ("GRU + LB", "GRU", "lb"), ("GRU + LB-R", "GRU", "lbres"),
           ("LB only", "LBOnly", "leastload"), ("LB-only-R", "LBOnly", "leastloadres")]
    out = []
    for met, title, sc, f in [("load_var_util", "Variance of server utilization", 1, "{:.3f}"), ("reassign_rate", "Reassignment rate (\\%)", 100, "{:.1f}")]:
        out.append("\\multicolumn{8}{l}{\\textit{" + title + "}} \\\\")
        for lab, m, pol in rws:
            out.append(lab + " & " + " & ".join(fmt(mean(srv, s, dn, m, pol, cap, met, sc), f) for s, dn, cap, _ in SETS) + " \\\\")
        out.append("\\midrule")
    W("tab13_loadvar.tex", out[:-1])
    notes.append("LB override rate (allocation level, mean over runs) is in alloc/*.json: lb_override_rate")
    W("ci_max.txt", notes)

    # ---- Figures 1-6 (LB at limited capacities, direct without a limit)
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = ["Observed"] + DL
    STY = {"Observed": ("#2a78d6", "o"), "DNN": ("#eb6834", "s"), "LSTM": ("#1baf9a", "^"), "GRU": ("#eda100", "D"),
           "Transformer": ("#e87ba4", "v"), "TFT": ("#118a2e", "P"), "Autoformer": ("#6250d6", "X"), "FEDformer": ("#e03c3c", "p")}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True, "pdf.fonttype": 42})
    lat = pd.read_csv(a.latency).set_index("model").latency_b1_median_ms if os.path.exists(a.latency) else None
    real = net[net.scen == "real"]
    def runs(m, cap, met, total=False):
        pol = "direct" if cap == 0 else "lb"
        x = real[(real.model == m) & (real.cap == cap) & (real.pol == pol)].sort_values("run")[met].values.astype(float)
        return x + lat[m] if (total and m != "Observed") else x
    def mci(x):
        return x.mean(), ci(x)
    CAPS = [(1, "1"), (5, "5"), (10, "10"), (40, "40"), (0, "No limit")]
    def overview(met, ylab, fn, total=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.1)); xs = np.arange(len(CAPS))
        for i, m in enumerate(M):
            c, mk = STY[m]; v = [mci(runs(m, cap, met, total)) for cap, _ in CAPS]
            ax.errorbar(xs + (i - 3.5) * 0.012, [p for p, _ in v], yerr=[q for _, q in v], color=c, marker=mk, ms=5, lw=1.3, capsize=3, label=m, mec="white", mew=0.6)
        ax.set_xticks(xs); ax.set_xticklabels([l_ for _, l_ in CAPS]); ax.set_xlabel("Server capacity (Mbps)"); ax.set_ylabel(ylab)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, fn + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(a.out, fn + ".png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    def zoom(met, ylab, fn, caps, f, total=False):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.74))
        for ax, (cap, title) in zip(axs, caps):
            vals = []
            for i, m in enumerate(M):
                c, mk = STY[m]; p, q = mci(runs(m, cap, met, total)); vals.append((p, q))
                ax.errorbar([i], [p], yerr=[q], color=c, marker=mk, ms=6, capsize=3, lw=1.2, mec="white", mew=0.6)
            lo = min(p - q for p, q in vals); hi = max(p + q for p, q in vals); pad = (hi - lo) * 0.15 or 0.01
            for i, (p, q) in enumerate(vals):
                ax.text(i, p + q + pad * 0.4, f.format(p), ha="center", va="bottom", fontsize=7, color="#52514e")
            ax.set_ylim(lo - pad, hi + pad * 1.6); ax.set_xticks(range(len(M))); ax.set_xticklabels(M, rotation=45, ha="right", fontsize=9); ax.set_title(title, fontsize=10)
        axs[0].set_ylabel(ylab); fig.tight_layout(); fig.savefig(os.path.join(a.out, fn + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(a.out, fn + ".png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    overview("goodput_mbps", "Goodput (Mbps)", "GoodputOverview")
    zoom("goodput_mbps", "Goodput (Mbps)", "GoodputZoom", [(5, "5 Mbps"), (1, "1 Mbps")], "{:.3f}")
    overview("delay_mean_ms", "Mean one-way delay (ms)", "DelayOverview")
    zoom("delay_mean_ms", "Mean one-way delay (ms)", "DelayZoom", [(40, "40 Mbps"), (1, "1 Mbps")], "{:.2f}")
    if lat is not None:
        overview("delay_mean_ms", "Total latency (ms)", "TotalLatencyOverview", total=True)
        zoom("delay_mean_ms", "Total latency (ms)", "TotalLatencyZoom", [(0, "No limit"), (1, "1 Mbps")], "{:.2f}", total=True)
    else:
        print("latency_CPU.csv not found: total-latency figures skipped")
    print("written to", a.out)


if __name__ == "__main__":
    main()
