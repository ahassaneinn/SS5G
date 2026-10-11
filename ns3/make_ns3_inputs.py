#!/usr/bin/env python3
"""
make_ns3_inputs.py -- ns-3 stage 1: build ns3_inputs/ from the outputs of ../python (run once).

Builds the inputs of the trace-driven ns-3 evaluation from the outputs of eua_pipeline.py:
  work/ns3_records.csv.gz   every trace record, oriented user<->server, real timestamps
  work/steps.csv.gz         one row per active user-second (serving server = max-bytes server)
  work/servers.csv          server IP, lat, lon, organisation
  results/predictions/      pred_<Model>_<fs>_seed<k>.csv.gz (TEST period only, top-5 ranked servers)

Design
  * every allocator replays the SAME traffic (test-period user-seconds of all users)
  * real per-second timing of the trace
  * only TEST-period decisions are simulated

Outputs (ns3_inputs/):
  steps.csv.gz      User,t,app,label,last,est_bps     one row per decision (test user-second)
  traffic.csv.gz    User,t,dir,proto,bytes            dominant-server traffic of each decision second
  servers.csv       ip,lat,lon,org
  users.csv         User,lat,lon,app
  decisions/<Name>__s<seed>.csv.gz   User,t,c1..c5    ranked candidate servers per decision
  manifest.json
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

GAMING = {"LOL", "TFT", "VAL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-dir", dest="python_dir", default=os.path.join("..", "python"),
                    help="folder of eua_pipeline.py with work/ and results/ (default ../python)")
    ap.add_argument("--out", default="ns3_inputs")
    ap.add_argument("--fs", default="full", help="feature set of the DL predictions to use")
    a = ap.parse_args()
    work = os.path.join(a.python_dir, "work")
    pdir = os.path.join(a.python_dir, "results", "predictions")
    os.makedirs(os.path.join(a.out, "decisions"), exist_ok=True)

    steps = pd.read_csv(os.path.join(work, "steps.csv.gz"))
    srv = pd.read_csv(os.path.join(work, "servers.csv"))
    rec = pd.read_csv(os.path.join(work, "ns3_records.csv.gz"))

    # ---------- decisions = test samples (identical for every prediction file)
    ref = pd.read_csv(os.path.join(pdir, "pred_Persistence_-_seed0.csv.gz"))
    key = ref[["User", "k", "t_label", "label_server"]].copy()
    st = steps.set_index(["User", "k"])
    last = st.loc[list(zip(key.User, key.k))]
    key["last"] = last.server.values
    key["est_bps"] = ((last.bytes_dl.values + last.bytes_ul.values) * 8.0)   # rate of the last observed second
    nxt = st.loc[list(zip(key.User, key.k + 1))]
    assert (nxt.t.values == key.t_label.values).all(), "t_label mismatch with steps"
    assert (nxt.server.values == key.label_server.values).all(), "label mismatch with steps"
    key["app"] = nxt.app.values
    out_steps = key.rename(columns={"t_label": "t", "label_server": "label"})[
        ["User", "t", "app", "label", "last", "est_bps"]]
    out_steps.to_csv(os.path.join(a.out, "steps.csv.gz"), index=False)

    # ---------- traffic of those seconds: records to the dominant (= label) server only
    r = rec.merge(out_steps[["User", "t", "label"]], on=["User", "t"], how="inner")
    tot_bytes = r.Length.sum()
    r = r[r.server == r.label]
    dom_share = r.Length.sum() / tot_bytes
    r["proto"] = np.where(r.transport == "TCP", "TCP", "UDP")          # 'OTHER' (28 rows) -> UDP
    tr = (r.groupby(["User", "t", "dir", "proto"]).Length.sum().reset_index()
          .rename(columns={"Length": "bytes"}).sort_values(["User", "t", "dir", "proto"]))
    tr.to_csv(os.path.join(a.out, "traffic.csv.gz"), index=False)

    # ---------- servers / users
    srv.rename(columns={"server": "ip", "srv_lat": "lat", "srv_lon": "lon", "srv_org": "org"})[
        ["ip", "lat", "lon", "org"]].to_csv(os.path.join(a.out, "servers.csv"), index=False)
    users = steps.groupby("User").agg(lat=("user_lat", "median"), lon=("user_lon", "median"),
                                      app=("app", "first")).reset_index()
    users.to_csv(os.path.join(a.out, "users.csv"), index=False)
    known = set(srv.server)

    # ---------- decision sets
    def write_dec(name, seed, cands):
        d = out_steps[["User", "t"]].copy()
        for j in range(5):
            d[f"c{j + 1}"] = [c[j] if j < len(c) else "" for c in cands]
        d.to_csv(os.path.join(a.out, "decisions", f"{name}__s{seed}.csv.gz"), index=False)

    def ranked(p):
        cols = ["pred_server", "pred2", "pred3", "pred4", "pred5"]
        cols = [c for c in cols if c in p.columns]
        lst = []
        for row, lastsrv in zip(p[cols].itertuples(index=False), out_steps["last"]):
            c = [s for s in row if isinstance(s, str) and s != "UNK" and s in known]
            c = list(dict.fromkeys(c))                  # unique, keep order
            if not c:
                c = [lastsrv]                           # model gave only unseen/UNK -> keep current server
            lst.append(c)
        return lst

    made = []
    # historical assignment (Observed)
    write_dec("Observed", 0, [[s] for s in out_steps.label]); made.append("Observed__s0")
    # persistence from the RAW current server (the pred file holds UNK for servers unseen in training)
    write_dec("Persistence", 0, [[s] for s in out_steps["last"]]); made.append("Persistence__s0")
    stats = {}
    for f in sorted(glob.glob(os.path.join(pdir, "pred_*.csv.gz"))):
        m = re.match(r"pred_(.+?)_(.+?)_seed(\d+)\.csv\.gz", os.path.basename(f))
        model, fs, seed = m.group(1), m.group(2), int(m.group(3))
        if model == "Persistence":
            continue
        if fs not in (a.fs, "-"):
            continue
        p = pd.read_csv(f)
        assert (p.User.values == out_steps.User.values).all() and (p.t_label.values == out_steps.t.values).all()
        c = ranked(p)
        write_dec(model, seed, c)
        made.append(f"{model}__s{seed}")
        stats[f"{model}__s{seed}"] = float(np.mean([x[0] == y for x, y in zip(c, out_steps.label)]))

    man = {
        "decisions": made,
        "n_decisions": int(len(out_steps)),
        "users": int(out_steps.User.nunique()),
        "servers_in_labels": int(out_steps.label.nunique()),
        "traffic_rows": int(len(tr)),
        "traffic_GB": float(tr.bytes.sum() / 1e9),
        "dominant_server_byte_share": float(dom_share),
        "udp_byte_share": float(tr[tr.proto == "UDP"].bytes.sum() / tr.bytes.sum()),
        "top1_acc_on_ns3_decisions": stats,
        "apps": out_steps.app.value_counts().to_dict(),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(json.dumps({k: v for k, v in man.items() if k != "top1_acc_on_ns3_decisions"}, indent=2))


if __name__ == "__main__":
    main()
