#!/usr/bin/env python
"""
eua_pipeline.py — deep learning-based edge user allocation (EUA) on the GAViST5G dataset.

Task definition:
  * Granularity: per user, per active second. The "serving server" of a second
    is the server endpoint that carried the most bytes for that user in that
    second (both directions summed).
  * Label: the serving server of the NEXT active second (t+1).
  * Inputs: only information available before that decision, i.e. the user's
    history up to and including second t (window of W steps), the user's
    location, and the requested application. No user ID, no absolute
    timestamp, nothing from second t+1.
  * Split: chronological per user (70/15/15 by default) with a purge of W
    samples at each boundary so no test window overlaps a training window.
    Scaler and vocabularies are fit on the training period only.
  * Second protocol: unseen users (5 user-disjoint folds, `holdout`).

Sub-commands (run `python eua_pipeline.py <cmd> -h` for options):
  prepare     raw CSV -> per-second step table + ns-3 record export
  smoke       2-minute check that every model builds/trains on your TF install
  baselines   persistence / user-majority / app-majority / nearest-server
  train       train DL models over seeds and feature sets, save metrics/preds
  holdout     unseen-user generalisation (GroupKFold over users)
  latency     inference latency (run once on GPU and once with CPU only)
  tune        optional Bayesian optimisation with an equal budget per model
  report      aggregate everything into CSV / Markdown / LaTeX tables

TensorFlow is imported lazily, so `prepare`, `baselines` and `report` run
without it. Input: the Canadian and international CSV files of GAViST5G
(see ../data/README.md).
"""
import argparse
import ipaddress
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TCP_FAMILY = {"TCP", "TLSv1", "TLSv1.1", "TLSv1.2", "TLSv1.3", "HTTP", "HTTP/JSON",
              "HTTP2", "SSL", "SSLv2", "SSLv3", "OCSP", "THRIFT"}
OTHER_FAMILY = {"ICMP", "ICMPv6", "ARP"}

# Canadian portion (CombinedALLGeo.csv): users are stored in contiguous blocks per app.
CANADA_APP_RANGES = [("Crave", 1, 10), ("LOL", 11, 19), ("Netflix", 20, 31),
                     ("Prime", 32, 42), ("TFT", 43, 50), ("VAL", 51, 54),
                     ("YouTube", 55, 83)]
APP_TYPE = {"Crave": "video", "Netflix": "video", "Prime": "video", "YouTube": "video",
            "LOL": "gaming", "TFT": "gaming", "VAL": "gaming"}

NUM_BASE = ["is_pad", "log_gap", "log_bytes_dl", "log_bytes_ul", "n_servers",
            "log_n_records", "udp_share", "artt", "user_lat", "user_lon", "is_hotspot"]
NUM_SRV = ["srv_lat", "srv_lon", "log_dist_km"]

FEATURE_SETS = {
    # name: (numeric columns, use server-id/org embeddings)
    "full":   ([c for c in NUM_BASE] + NUM_SRV, True),
    "no_rtt": ([c for c in NUM_BASE if c != "artt"] + NUM_SRV, True),
    "no_srv": ([c for c in NUM_BASE], False),       # no server history at all
}
ALL_MODELS = ["DNN", "LSTM", "GRU", "Transformer", "Autoformer", "TFT", "FEDformer"]
PAD, UNK = 0, 1


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def is_private(ip):
    try:
        return ipaddress.ip_address(str(ip)).is_private
    except ValueError:
        return False


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def transport_of(p):
    p = str(p)
    if p in TCP_FAMILY or p.startswith("TLS") or p.startswith("HTTP"):
        return "TCP"
    if p in OTHER_FAMILY:
        return "OTHER"
    return "UDP"


# --------------------------------------------------------------------------
# 1. PREPARE
# --------------------------------------------------------------------------
def detect_user_ips(df):
    """Per user: the set of IPs that are the user's own endpoint.
    Private IPs are always user-side. Users with no private IP (e.g. public
    IPv6 on the 5G modem) get the endpoint with the highest row coverage and
    the most distinct peers."""
    out = {}
    for u, g in df.groupby("User"):
        eps = pd.unique(pd.concat([g.SourceIP, g.DestinationIP]))
        priv = {ip for ip in eps if is_private(ip)}
        rest = g[~(g.SourceIP.isin(priv) | g.DestinationIP.isin(priv))]
        out[u] = set(priv)
        if len(rest) < 0.01 * len(g):
            continue
        # rows with no private endpoint (e.g. public IPv6 on the 5G modem):
        # the user endpoint is the IP with the most distinct peers / coverage
        cov = Counter()
        peers = defaultdict(set)
        for s, d in zip(rest.SourceIP, rest.DestinationIP):
            cov[s] += 1
            cov[d] += 1
            peers[s].add(d)
            peers[d].add(s)
        best = max(cov, key=lambda ip: (len(peers[ip]), cov[ip]))
        out[u].add(best)
    return out


def assign_apps(users, data_path, single_app=None, app_map=None):
    if app_map:
        m = pd.read_csv(app_map)
        return dict(zip(m.User, m.app))
    if single_app:
        return {u: single_app for u in users}
    if "CombinedALLGeo" in os.path.basename(data_path):
        res = {}
        for app, a, b in CANADA_APP_RANGES:
            for u in range(a, b + 1):
                res[u] = app
        missing = [u for u in users if u not in res]
        if missing:
            raise SystemExit(f"Users {missing} not in the Canadian app map; pass --app-map")
        return res
    raise SystemExit("Cannot infer the application per user: pass --single-app or --app-map")


def cmd_prepare(a):
    log("Reading", a.data)
    df = pd.read_csv(a.data, low_memory=False)
    need = ["AdjustedTime", "SourceIP", "DestinationIP", "Protocol", "SourceOrg",
            "SourceLongitude", "SourceLatitude", "Connection", "User", "DestinationOrg",
            "DestinationLongitude", "DestinationLatitude", "Length", "ARTT"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns {miss}. Use the *Geo.csv files (CombinedAggUserGeo).")
    n0 = len(df)
    df["t_abs"] = pd.to_datetime(df.AdjustedTime)
    t0 = df.t_abs.min()
    df["t"] = ((df.t_abs - t0).dt.total_seconds()).astype(np.int64)

    uips = detect_user_ips(df)
    src_u = np.array([s in uips[u] for s, u in zip(df.SourceIP, df.User)])
    dst_u = np.array([d in uips[u] for d, u in zip(df.DestinationIP, df.User)])
    keep = src_u ^ dst_u
    log(f"rows {n0}; user->server {int((src_u & ~dst_u).sum())}, server->user {int((~src_u & dst_u).sum())}, "
        f"dropped (no/both user endpoint) {int((~keep).sum())}")
    df = df[keep].copy()
    ul = src_u[keep]
    df["dir"] = np.where(ul, "UL", "DL")
    df["user_ip"] = np.where(ul, df.SourceIP, df.DestinationIP)
    df["server"] = np.where(ul, df.DestinationIP, df.SourceIP)
    df["srv_lat"] = np.where(ul, df.DestinationLatitude, df.SourceLatitude).astype(float)
    df["srv_lon"] = np.where(ul, df.DestinationLongitude, df.SourceLongitude).astype(float)
    df["srv_org"] = np.where(ul, df.DestinationOrg, df.SourceOrg)
    ulat = np.where(ul, df.SourceLatitude, df.DestinationLatitude).astype(float)
    ulon = np.where(ul, df.SourceLongitude, df.DestinationLongitude).astype(float)
    # Anonymised user geo (private IPs) was written as (Longitude=lat, Latitude=lon): swap back.
    swap = np.array([is_private(ip) for ip in df.user_ip]) | (np.abs(ulat) > 90)
    df["user_lat"] = np.where(swap, ulon, ulat)
    df["user_lon"] = np.where(swap, ulat, ulon)
    # one location per (stationary) user: prefer the anonymised device location
    df["_priv"] = [is_private(ip) for ip in df.user_ip]
    loc_p = df[df._priv].groupby("User")[["user_lat", "user_lon"]].median()
    loc_a = df.groupby("User")[["user_lat", "user_lon"]].median()
    loc = loc_a.copy()
    loc.loc[loc_p.index] = loc_p
    df["user_lat"] = df.User.map(loc.user_lat)
    df["user_lon"] = df.User.map(loc.user_lon)
    bad = df.srv_lat.isna() | df.srv_lon.isna() | df.user_lat.isna()
    log(f"dropping {int(bad.sum())} rows with missing server/user geolocation")
    df = df[~bad].copy()
    df["srv_org"] = df.srv_org.fillna("Unknown")
    df["transport"] = df.Protocol.map(transport_of)
    df["Length"] = df.Length.astype(np.int64)

    apps = assign_apps(sorted(df.User.unique()), a.data, a.single_app, a.app_map)
    df["app"] = df.User.map(apps)

    os.makedirs(a.work, exist_ok=True)
    # ---- server table
    srv = (df.groupby("server").agg(srv_lat=("srv_lat", "first"), srv_lon=("srv_lon", "first"),
                                     srv_org=("srv_org", "first"), rows=("Length", "size"))
           .reset_index())
    srv.to_csv(os.path.join(a.work, "servers.csv"), index=False)

    # ---- per (user, t, server)
    df["hot"] = (df.Connection == "5GHotspot").astype(float)
    df["b_dl"] = np.where(df.dir == "DL", df.Length, 0)
    df["b_ul"] = np.where(df.dir == "UL", df.Length, 0)
    df["b_udp"] = np.where(df.transport == "UDP", df.Length, 0)
    uts = (df.groupby(["User", "t", "server"], sort=False)
           .agg(bytes=("Length", "sum")).reset_index())
    uts = uts.sort_values(["User", "t", "bytes", "server"], ascending=[True, True, False, True])
    dom = uts.drop_duplicates(["User", "t"])[["User", "t", "server"]]
    per = (df.groupby(["User", "t"])
           .agg(bytes_dl=("b_dl", "sum"), bytes_ul=("b_ul", "sum"), b_udp=("b_udp", "sum"),
                bytes=("Length", "sum"), n_records=("Length", "size"),
                n_servers=("server", "nunique"), artt=("ARTT", "mean"),
                user_lat=("user_lat", "median"), user_lon=("user_lon", "median"),
                is_hotspot=("hot", "mean"),
                app=("app", "first"))
           .reset_index())
    steps = per.merge(dom, on=["User", "t"], how="left")
    steps = steps.merge(srv[["server", "srv_lat", "srv_lon", "srv_org"]], on="server", how="left")
    steps["udp_share"] = steps.b_udp / steps.bytes.clip(lower=1)
    steps["dist_km"] = haversine_km(steps.user_lat, steps.user_lon, steps.srv_lat, steps.srv_lon)
    steps = steps.sort_values(["User", "t"]).reset_index(drop=True)
    steps["gap"] = steps.groupby("User").t.diff().fillna(0).clip(lower=0)
    steps["session"] = (steps.gap > a.session_gap).astype(int).groupby(steps.User).cumsum()
    steps["k"] = steps.groupby("User").cumcount()
    steps.drop(columns=["b_udp", "bytes"]).to_csv(os.path.join(a.work, "steps.csv.gz"), index=False)

    # ---- ns-3 export: every record, with the dominant-flow flag (input of ns3/make_ns3_inputs.py)
    rec = df[["User", "t", "user_ip", "server", "dir", "transport", "Length", "app",
              "user_lat", "user_lon"]].copy()
    rec = rec.merge(dom.rename(columns={"server": "dom_server"}), on=["User", "t"], how="left")
    rec["is_dominant"] = (rec.server == rec.dom_server).astype(int)
    rec.drop(columns=["dom_server"]).to_csv(os.path.join(a.work, "ns3_records.csv.gz"), index=False)

    # ---- dataset statistics
    stats = {
        "source_file": os.path.basename(a.data), "raw_rows": int(n0), "kept_rows": int(len(df)),
        "users": int(df.User.nunique()), "servers": int(srv.shape[0]),
        "apps": {k: int(v) for k, v in pd.Series(apps).value_counts().items()},
        "user_seconds": int(len(steps)), "sessions": int(steps.groupby("User").session.max().add(1).sum()),
        "session_gap_s": a.session_gap,
        "persistence_rate_next_second": float(
            (steps.server.values[1:] == steps.server.values[:-1])[steps.User.values[1:] == steps.User.values[:-1]].mean()),
        "per_user_rate_mbps_quantiles": {str(q): float(v) for q, v in
            ((df.groupby(["User", "t"]).Length.sum() * 8 / 1e6).quantile([.5, .9, .95, .99])).items()},
        "per_server_concurrent_rate_mbps_quantiles": {str(q): float(v) for q, v in
            ((df.groupby(["server", "t"]).Length.sum() * 8 / 1e6).quantile([.5, .9, .95, .99])).items()},
        "per_app_user_rate_mbps_median": {k: float(v) for k, v in
            (df.groupby(["app", "User", "t"]).Length.sum() * 8 / 1e6).groupby(level=0).median().items()},
        "udp_byte_share": float(df.b_udp.sum() / df.Length.sum()),
        "artt_ms_mean": float(df.ARTT.mean()),
    }
    with open(os.path.join(a.work, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log("Saved steps.csv.gz, servers.csv, ns3_records.csv.gz, dataset_stats.json to", a.work)
    log(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------
# 2. DATASET (windows, splits, vocab, scaling)
# --------------------------------------------------------------------------
class Dataset:
    """Builds windows from steps.csv.gz.

    sample (u, k): inputs = steps k-W+1..k of user u (left-padded), label = server at k+1.
    """

    def __init__(self, work, W=60, split="chrono", fracs=(0.70, 0.15, 0.15), purge=None,
                 test_users=None, val_frac_users=0.15, max_users=None):
        self.W = W
        self._work = work
        self._cache = {}
        st = pd.read_csv(os.path.join(work, "steps.csv.gz"))
        if max_users:
            keep = sorted(st.User.unique())[:max_users]
            st = st[st.User.isin(keep)]
        self.steps = st.reset_index(drop=True)
        purge = W if purge is None else purge
        # ---- sample table
        st = self.steps
        nxt_same = st.User.values[1:] == st.User.values[:-1]
        idx = np.nonzero(nxt_same)[0]                 # row index of the input's last step
        S = pd.DataFrame({"row": idx, "User": st.User.values[idx], "k": st.k.values[idx],
                          "label_server": st.server.values[idx + 1], "last_server": st.server.values[idx],
                          "t_label": st.t.values[idx + 1], "app": st.app.values[idx]})
        n_per = S.groupby("User").k.transform("max") + 1
        part = np.full(len(S), "drop", dtype=object)
        if split == "chrono":
            a, b = fracs[0], fracs[0] + fracs[1]
            tr_end = np.floor(a * n_per).astype(int)
            va_end = np.floor(b * n_per).astype(int)
            k = S.k.values
            part[k < tr_end] = "train"
            part[(k >= tr_end + purge) & (k < va_end)] = "val"
            part[k >= va_end + purge] = "test"
        elif split == "users":
            test_users = set(test_users)
            is_test = S.User.isin(test_users).values
            part[is_test] = "test"
            tr_end = np.floor((1 - val_frac_users) * n_per).astype(int)
            k = S.k.values
            part[~is_test & (k < tr_end)] = "train"
            part[~is_test & (k >= tr_end + purge)] = "val"
        S["part"] = part
        self.S = S[S.part != "drop"].reset_index(drop=True)

        # ---- vocabularies from the TRAINING period only
        tr_rows = self.S[self.S.part == "train"]
        last_train_k = tr_rows.groupby("User").k.max()
        in_train_period = st.User.map(last_train_k).fillna(-2).values + 1 >= st.k.values  # inputs+label (users without train samples excluded)
        srv_train = pd.unique(st.server.values[in_train_period])
        self.srv_vocab = {s: i + 2 for i, s in enumerate(sorted(srv_train))}
        self.n_srv = len(self.srv_vocab) + 2
        orgs = pd.unique(st.srv_org.fillna("Unknown").values[in_train_period])
        self.org_vocab = {o: i + 2 for i, o in enumerate(sorted(orgs))}
        self.n_org = len(self.org_vocab) + 2
        self.app_vocab = {a_: i + 2 for i, a_ in enumerate(sorted(st.app.unique()))}
        self.n_app = len(self.app_vocab) + 2
        self.idx2srv = {i: s for s, i in self.srv_vocab.items()}
        self.srv_org = dict(zip(st.server, st.srv_org.fillna("Unknown")))
        self.train_label_counts = Counter(self.enc_srv(tr_rows.label_server.values))

        # ---- per-step numeric matrix (all candidate columns) + scaler fit on train period
        st = st.copy()
        st["is_pad"] = 0.0
        st["log_gap"] = np.log1p(st.gap)
        st["log_bytes_dl"] = np.log1p(st.bytes_dl)
        st["log_bytes_ul"] = np.log1p(st.bytes_ul)
        st["log_n_records"] = np.log1p(st.n_records)
        st["log_dist_km"] = np.log1p(st.dist_km)
        self.all_num = NUM_BASE + NUM_SRV
        X = st[self.all_num].astype(np.float32).values
        mu = X[in_train_period].mean(0)
        sd = X[in_train_period].std(0) + 1e-6
        mu[0], sd[0] = 0.0, 1.0                       # keep is_pad as a 0/1 flag
        self.Xnum = ((X - mu) / sd).astype(np.float32)
        self.srv_idx = self.enc_srv(st.server.values)
        self.org_idx = np.array([self.org_vocab.get(o, UNK) for o in st.srv_org.fillna("Unknown")], np.int32)
        self.app_idx = np.array([self.app_vocab[x] for x in st.app], np.int32)
        self.user_start = st.groupby("User").apply(lambda g: g.index.min()).to_dict()

    def enc_srv(self, arr):
        return np.array([self.srv_vocab.get(s, UNK) for s in arr], np.int32)

    def windows(self, part, feature_set="full"):
        """Returns (inputs list, y, meta DataFrame) for one partition."""
        key = (part, feature_set)
        if key in self._cache:
            return self._cache[key]
        if any(k[1] != feature_set for k in self._cache):
            self._cache = {}                            # keep memory bounded: one feature set at a time
        cols, use_srv = FEATURE_SETS[feature_set]
        ci = [self.all_num.index(c) for c in cols]
        S = self.S[self.S.part == part]
        rows = S.row.values
        starts = np.array([self.user_start[u] for u in S.User.values])
        W = self.W
        offs = np.arange(-W + 1, 1)
        R = rows[:, None] + offs[None, :]
        pad = R < starts[:, None]
        Rc = np.where(pad, rows[:, None], R)
        Xn = self.Xnum[Rc][:, :, ci].copy()
        Xn[pad] = 0.0
        if "is_pad" in cols:
            Xn[:, :, cols.index("is_pad")] = pad.astype(np.float32)
        app = np.where(pad, PAD, self.app_idx[Rc]).astype(np.int32)
        inputs = {"num": Xn, "app": app}
        if use_srv:
            inputs["srv"] = np.where(pad, PAD, self.srv_idx[Rc]).astype(np.int32)
            inputs["org"] = np.where(pad, PAD, self.org_idx[Rc]).astype(np.int32)
        y = self.enc_srv(S.label_server.values)
        meta = S[["User", "k", "t_label", "label_server", "last_server", "app"]].reset_index(drop=True)
        meta["last_idx"] = self.enc_srv(S.last_server.values)
        self._cache[key] = (inputs, y, meta)
        return inputs, y, meta

    def n_classes(self):
        return self.n_srv


# --------------------------------------------------------------------------
# 3. METRICS
# --------------------------------------------------------------------------
def ece_score(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def compute_metrics(y, topk, meta, ds, probs=None, prefix=""):
    """y: (n,) true class idx (UNK=1 for servers unseen in training);
    topk: (n,K) ranked predictions (K>=1). probs optional (n,C)."""
    from sklearn.metrics import f1_score, recall_score
    pred = topk[:, 0]
    ok = (pred == y) & (y != UNK)
    m = {}
    m["n_test"] = int(len(y))
    m["acc"] = float(ok.mean())
    for K in (3, 5):
        if topk.shape[1] >= K:
            m[f"top{K}"] = float(((topk[:, :K] == y[:, None]).any(1) & (y != UNK)).mean())
        else:
            m[f"top{K}"] = np.nan
    labels = np.unique(y[y != UNK])
    yy = np.where(y == UNK, -1, y)
    m["macro_f1"] = float(f1_score(yy, pred, labels=labels, average="macro", zero_division=0))
    m["weighted_f1"] = float(f1_score(yy, pred, labels=labels, average="weighted", zero_division=0))
    m["balanced_acc"] = float(recall_score(yy, pred, labels=labels, average="macro", zero_division=0))
    change = meta.last_idx.values != y
    m["change_rate"] = float(change.mean())
    m["acc_on_change"] = float(ok[change].mean()) if change.any() else np.nan
    m["acc_on_stay"] = float(ok[~change].mean()) if (~change).any() else np.nan
    m["unseen_label_rate"] = float((y == UNK).mean())
    # organisation-level (provider) accuracy
    org_true = np.array([ds.srv_org.get(ds.idx2srv.get(i, ""), "UNK") for i in y])
    org_pred = np.array([ds.srv_org.get(ds.idx2srv.get(i, ""), "UNK?") for i in pred])
    m["org_acc"] = float((org_true == org_pred).mean())
    # distance error between predicted and true server (km)
    # rare-class analysis by training frequency of the true label
    cnt = np.array([ds.train_label_counts.get(i, 0) for i in y])
    bins = [("head_>=1000", cnt >= 1000), ("mid_100-999", (cnt >= 100) & (cnt < 1000)),
            ("tail_10-99", (cnt >= 10) & (cnt < 100)), ("rare_1-9", (cnt >= 1) & (cnt < 10)),
            ("unseen_0", cnt == 0)]
    for name, msk in bins:
        m[f"share_{name}"] = float(msk.mean())
        m[f"recall_{name}"] = float(ok[msk].mean()) if msk.any() else np.nan
    if probs is not None:
        p = probs[np.arange(len(y)), np.clip(y, 0, probs.shape[1] - 1)]
        p = np.where(y == UNK, 1e-12, p)
        m["nll"] = float(-np.log(np.clip(p, 1e-12, 1)).mean())
        m["ece"] = ece_score(probs.max(1), ok)
    else:
        m["nll"] = np.nan
        m["ece"] = np.nan
    if prefix:
        m = {prefix + k: v for k, v in m.items()}
    return m


# --------------------------------------------------------------------------
# 4. BASELINES
# --------------------------------------------------------------------------
def run_baselines(ds, part="test"):
    _, y, meta = ds.windows(part, "no_srv")
    tr = ds.S[ds.S.part == "train"]
    tr_y = ds.enc_srv(tr.label_server.values)
    K = 5
    # frequency rankings
    user_rank = {u: [s for s, _ in Counter(tr_y[tr.User.values == u]).most_common(K)] for u in tr.User.unique()}
    app_rank = {a_: [s for s, _ in Counter(tr_y[tr.app.values == a_]).most_common(K)] for a_ in tr.app.unique()}
    glob_rank = [s for s, _ in Counter(tr_y).most_common(K)]

    def pad_rank(lst):
        lst = list(dict.fromkeys(lst))
        for s in glob_rank:
            if len(lst) >= K:
                break
            if s not in lst:
                lst.append(s)
        while len(lst) < K:
            lst.append(PAD)
        return lst[:K]

    out = {}
    # persistence: next serving server = current serving server
    tk = np.array([pad_rank([l] + user_rank.get(u, [])) for l, u in zip(meta.last_idx, meta.User)])
    out["Persistence"] = tk
    out["UserMajority"] = np.array([pad_rank(user_rank.get(u, [])) for u in meta.User])
    out["AppMajority"] = np.array([pad_rank(app_rank.get(a_, [])) for a_ in meta.app])
    # nearest server among the app's training servers (geographic heuristic)
    srv = pd.read_csv(os.path.join(ds._work, "servers.csv")).set_index("server")
    st = ds.steps
    uloc = st.groupby("User")[["user_lat", "user_lon"]].median()
    app_cands = {a_: sorted(set(tr_y[tr.app.values == a_]) - {UNK}) for a_ in tr.app.unique()}
    near = {}
    for u in meta.User.unique():
        a_ = st.loc[st.User == u, "app"].iloc[0]
        c = np.array(app_cands.get(a_, []))
        if len(c) == 0:
            near[u] = [PAD] * K
            continue
        ips = [ds.idx2srv[i] for i in c]
        la = srv.loc[ips, "srv_lat"].values
        lo = srv.loc[ips, "srv_lon"].values
        d = haversine_km(uloc.loc[u, "user_lat"], uloc.loc[u, "user_lon"], la, lo)
        near[u] = pad_rank(list(c[np.argsort(d, kind="stable")][:K]))
    out["NearestServer"] = np.array([near[u] for u in meta.User])
    return y, meta, out


def cmd_baselines(a):
    os.makedirs(a.out, exist_ok=True)
    ds = Dataset(a.work, W=a.window, max_users=a.max_users)
    log(f"samples: train {sum(ds.S.part == 'train')}, val {sum(ds.S.part == 'val')}, "
        f"test {sum(ds.S.part == 'test')}; classes {ds.n_srv}")
    y, meta, res = run_baselines(ds)
    rows = []
    for name, tk in res.items():
        m = compute_metrics(y, tk, meta, ds)
        m.update(model=name, feature_set="-", seed=0, split="chrono")
        rows.append(m)
        log(f"{name:14s} acc={m['acc']:.4f} top5={m['top5']:.4f} macroF1={m['macro_f1']:.4f} "
            f"acc_on_change={m['acc_on_change']:.4f}")
        save_predictions(a.out, name, "-", 0, meta, tk, ds)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "metrics_baselines.csv"), index=False)


def save_predictions(out, model, fs, seed, meta, topk, ds, conf=None):
    d = os.path.join(out, "predictions")
    os.makedirs(d, exist_ok=True)
    p = meta[["User", "k", "t_label", "label_server"]].copy()
    p["pred_server"] = [ds.idx2srv.get(i, "UNK") for i in topk[:, 0]]
    for j in range(1, min(5, topk.shape[1])):
        p[f"pred{j + 1}"] = [ds.idx2srv.get(i, "UNK") for i in topk[:, j]]
    if conf is not None:
        p["conf"] = conf
    fn = f"pred_{model}_{fs}_seed{seed}.csv.gz".replace("/", "-")
    p.to_csv(os.path.join(d, fn), index=False)


# --------------------------------------------------------------------------
# 5. MODELS (TensorFlow / Keras; works with tf.keras 2.10+ and Keras 3)
# --------------------------------------------------------------------------
DEFAULT_CFG = {"units": 64, "dense": 32, "d_model": 32, "ff": 128, "heads": 4,
               "dropout": 0.2, "l2": 1e-3, "lr": 1e-3, "srv_emb": 16, "org_emb": 4, "app_emb": 4,
               "ma_kernel": 5}


def build_model(name, W, F, use_srv, n_srv, n_org, n_app, n_classes, cfg=None):
    import tensorflow as tf
    L = tf.keras.layers
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] else None

    class PositionalEncoding(L.Layer):
        def __init__(self, length, depth, **kw):
            super().__init__(**kw)
            pos = np.arange(length)[:, None]
            i = np.arange(depth)[None, :]
            ang = pos / np.power(10000, (2 * (i // 2)) / np.float32(depth))
            ang[:, 0::2] = np.sin(ang[:, 0::2])
            ang[:, 1::2] = np.cos(ang[:, 1::2])
            self.pe = tf.constant(ang[None], dtype=tf.float32)

        def call(self, x):
            return x + self.pe

    class EncoderBlock(L.Layer):
        """kind: 'transformer' | 'autoformer' (moving-average decomposition)
                 'fedformer' (decomposition + attention on the real FFT spectrum over time)
                 'tft' (sigmoid gating of inputs before attention)"""

        def __init__(self, kind, d, heads, ff, drop, k_ma, **kw):
            super().__init__(**kw)
            self.kind = kind
            self.att = L.MultiHeadAttention(num_heads=heads, key_dim=max(4, d // heads))
            self.n1 = L.LayerNormalization(epsilon=1e-6)
            self.n2 = L.LayerNormalization(epsilon=1e-6)
            self.d1 = L.Dropout(drop)
            self.d2 = L.Dropout(drop)
            self.f1 = L.Dense(ff, activation="relu")
            self.f2 = L.Dense(d)
            if kind in ("autoformer", "fedformer"):
                self.ma = L.AveragePooling1D(pool_size=k_ma, strides=1, padding="same")
            if kind == "tft":
                self.gate = L.Dense(d, activation="sigmoid")

        def call(self, x, training=None):
            if self.kind == "tft":
                x = x * self.gate(x)
            if self.kind in ("autoformer", "fedformer"):
                trend = self.ma(x)
                seas = x - trend
                if self.kind == "fedformer":
                    z = tf.transpose(seas, [0, 2, 1])
                    z = tf.math.real(tf.signal.fft(tf.cast(z, tf.complex64)))
                    q = tf.transpose(z, [0, 2, 1])
                else:
                    q = seas
                h = self.n1(seas + self.d1(self.att(q, q), training=training))
                h2 = self.f2(self.f1(h))
                return self.n2(trend + h + self.d2(h2, training=training))
            h = self.n1(x + self.d1(self.att(x, x), training=training))
            return self.n2(h + self.d2(self.f2(self.f1(h)), training=training))

    num = L.Input((W, F), name="num")
    app = L.Input((W,), dtype="int32", name="app")
    ins = [num, app]
    parts = [num, L.Embedding(n_app, cfg["app_emb"], name="emb_app")(app)]
    if use_srv:
        srv = L.Input((W,), dtype="int32", name="srv")
        org = L.Input((W,), dtype="int32", name="org")
        ins += [srv, org]
        parts += [L.Embedding(n_srv, cfg["srv_emb"], name="emb_srv")(srv),
                  L.Embedding(n_org, cfg["org_emb"], name="emb_org")(org)]
    x = L.Concatenate(axis=-1)(parts)
    u, dn, dm, drop = cfg["units"], cfg["dense"], cfg["d_model"], cfg["dropout"]

    if name == "DNN":                                   # non-temporal: last step only
        h = L.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
        h = L.Dense(u, activation="tanh", kernel_regularizer=reg)(h)
        h = L.Dense(max(8, u // 2), activation="tanh", kernel_regularizer=reg)(h)
        h = L.Dense(max(8, u // 4), activation="tanh", kernel_regularizer=reg)(h)
    elif name in ("LSTM", "GRU"):
        rnn = L.LSTM if name == "LSTM" else L.GRU
        h = rnn(u, activation="tanh")(x)
        h = L.Dropout(drop)(h)
        h = L.Dense(dn, activation="tanh")(h)
    else:
        kind = {"Transformer": "transformer", "Autoformer": "autoformer",
                "FEDformer": "fedformer", "TFT": "tft"}[name]
        n_blocks = 2 if name == "Transformer" else 3
        h = L.Dense(dm, name="proj")(x)
        if name in ("Transformer", "TFT"):
            h = PositionalEncoding(W, dm)(h)
        for i in range(n_blocks):
            h = EncoderBlock(kind, dm, cfg["heads"], cfg["ff"], 0.1, cfg["ma_kernel"], name=f"enc{i}")(h)
        h = L.GlobalAveragePooling1D()(h)
        h = L.Dense(u, activation="relu")(h)
        h = L.Dropout(drop)(h)
        h = L.Dense(dn, activation="relu")(h)
    out = L.Dense(n_classes, activation="softmax", name="out")(h)
    model = tf.keras.Model(ins, out, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    names = ["num", "app", "srv", "org"] if use_srv else ["num", "app"]
    return model, names


def as_list(inputs, names):
    return [inputs[n] for n in names]


def predict_topk(model, xs, K=5, batch=1024):
    probs = model.predict(xs, batch_size=batch, verbose=0)
    probs[:, PAD] = 0.0
    probs[:, UNK] = 0.0
    topk = np.argsort(-probs, axis=1)[:, :K]
    return probs, topk


def setup_tf(seed=None, quiet=True):
    if quiet:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    return tf


def train_one(ds, model_name, fs, seed, a, cfg=None, split_tag="chrono", save_dir=None):
    tf = setup_tf(seed)
    use_srv = FEATURE_SETS[fs][1]
    xtr, ytr, _ = ds.windows("train", fs)
    xva, yva, _ = ds.windows("val", fs)
    xte, yte, mte = ds.windows("test", fs)
    model, names = build_model(model_name, ds.W, xtr["num"].shape[2], use_srv,
                               ds.n_srv, ds.n_org, ds.n_app, ds.n_classes(), cfg)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(as_list(xtr, names), ytr, validation_data=(as_list(xva, names), yva),
                     epochs=a.epochs, batch_size=a.batch, callbacks=cbs, verbose=2 if a.verbose else 0,
                     shuffle=True)
    train_s = time.time() - t0
    n_ep = len(hist.history["loss"])
    probs, topk = predict_topk(model, as_list(xte, names))
    m = compute_metrics(yte, topk, mte, ds, probs)
    m.update(model=model_name, feature_set=fs, seed=seed, split=split_tag, epochs=n_ep,
             train_time_s=round(train_s, 1), time_per_epoch_s=round(train_s / max(1, n_ep), 2),
             params=int(model.count_params()), n_train=int(len(ytr)), n_val=int(len(yva)),
             best_val_loss=float(min(hist.history["val_loss"])))
    if save_dir:
        os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
        model.save_weights(os.path.join(save_dir, "weights", f"{model_name}_{fs}_seed{seed}.weights.h5"))
    tf.keras.backend.clear_session()
    return m, mte, topk, probs.max(1)


def append_row(path, row):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        if list(old.columns) != list(df.columns):     # rows with different metrics (e.g. baselines vs models)
            pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
            return
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done_set(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.model, d.feature_set, d.seed.astype(int)))


def load_cfg(a, model_name):
    if getattr(a, "use_tuned", False):
        p = os.path.join(a.out, "tuning", f"{model_name}.json")
        if os.path.exists(p):
            return json.load(open(p))["best"]
        log(f"WARNING: no tuned config for {model_name}; using defaults")
    return None


def env_report():
    tf = setup_tf()
    gpus = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow {tf.__version__}; GPUs: {[g.name for g in gpus] or 'NONE (training will be slow)'}")


def cmd_smoke(a):
    """2-minute end-to-end check of every model on a few users (run this first)."""
    env_report()
    a.epochs, a.batch, a.patience, a.verbose = 1, 128, 1, False
    ds = Dataset(a.work, W=20, max_users=6)
    for fs in ["full", "no_srv"]:
        for mname in ALL_MODELS:
            m, meta, topk, conf = train_one(ds, mname, fs, 0, a)
            log(f"OK {mname:12s} {fs:7s} acc={m['acc']:.3f} params={m['params']}")
    a2 = argparse.Namespace(**vars(a))
    a2.out = os.path.join(a.out, "_smoke")
    os.makedirs(a2.out, exist_ok=True)
    m, meta, topk, conf = train_one(ds, "GRU", "full", 0, a, save_dir=a2.out)
    save_predictions(a2.out, "GRU", "full", 0, meta, topk, ds, conf)
    log("Smoke test passed: all models build, train, predict and save.")


def cmd_train(a):
    env_report()
    os.makedirs(a.out, exist_ok=True)
    ds = Dataset(a.work, W=a.window, max_users=a.max_users)
    log(f"samples: train {sum(ds.S.part == 'train')}, val {sum(ds.S.part == 'val')}, "
        f"test {sum(ds.S.part == 'test')}; classes {ds.n_srv}")
    models = ALL_MODELS if a.models == ["all"] else a.models
    path = os.path.join(a.out, "metrics_models.csv")
    done = done_set(path)
    for fs in a.feature_sets:
        for mname in models:
            for seed in a.seeds:
                if (mname, fs, seed) in done:
                    log(f"skip {mname}/{fs}/seed{seed} (already in {path})")
                    continue
                log(f"=== {mname} | features={fs} | seed={seed}")
                m, meta, topk, conf = train_one(ds, mname, fs, seed, a, load_cfg(a, mname),
                                                save_dir=a.out if seed == a.seeds[0] else None)
                append_row(path, m)
                if fs == "full" or a.save_all_preds:
                    save_predictions(a.out, mname, fs, seed, meta, topk, ds, conf)
                log(f"    acc={m['acc']:.4f} top5={m['top5']:.4f} macroF1={m['macro_f1']:.4f} "
                    f"acc_on_change={m['acc_on_change']:.4f} ece={m['ece']:.4f} "
                    f"epochs={m['epochs']} time={m['train_time_s']}s")


# --------------------------------------------------------------------------
# 6. UNSEEN-USER HOLD-OUT
# --------------------------------------------------------------------------
def user_folds(work, k=5):
    st = pd.read_csv(os.path.join(work, "steps.csv.gz"), usecols=["User", "app"])
    ua = st.drop_duplicates("User").sort_values("User")
    folds = defaultdict(list)
    for app, g in ua.groupby("app"):
        for i, u in enumerate(g.User.values):
            folds[i % k].append(int(u))
    return [folds[i] for i in range(k)]


def cmd_holdout(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "metrics_holdout.csv")
    folds = user_folds(a.work, a.folds)
    json.dump(folds, open(os.path.join(a.out, "holdout_folds.json"), "w"))
    for fi, test_users in enumerate(folds):
        ds = Dataset(a.work, W=a.window, split="users", test_users=test_users)
        log(f"fold {fi}: test users {test_users}; train {sum(ds.S.part == 'train')} test {sum(ds.S.part == 'test')}")
        y, meta, res = run_baselines(ds)
        for name, tk in res.items():
            if name == "UserMajority":
                continue                                # undefined for unseen users
            m = compute_metrics(y, tk, meta, ds)
            m.update(model=name, feature_set="-", seed=0, fold=fi, split="unseen_users")
            append_row(path, m)
        if a.models:
            for mname in a.models:
                for fs in a.feature_sets:
                    m, *_ = train_one(ds, mname, fs, a.seed, a, split_tag="unseen_users")
                    m["fold"] = fi
                    append_row(path, m)
                    log(f"    {mname}/{fs} fold {fi}: acc={m['acc']:.4f} unseen_label_rate={m['unseen_label_rate']:.3f}")


# --------------------------------------------------------------------------
# 7. LATENCY
# --------------------------------------------------------------------------
def cmd_latency(a):
    tf = setup_tf(0)
    gpus = tf.config.list_physical_devices("GPU")
    device = "GPU" if gpus else "CPU"
    log("Measuring on", device, "| TF", tf.__version__)
    ds = Dataset(a.work, W=a.window, max_users=a.max_users)
    rows = []
    models = ALL_MODELS if a.models == ["all"] else a.models
    for mname in models:
        fs = "full"
        x, _, _ = ds.windows("test", fs)
        model, names = build_model(mname, ds.W, x["num"].shape[2], True, ds.n_srv, ds.n_org,
                                   ds.n_app, ds.n_classes(), load_cfg(a, mname))
        wp = os.path.join(a.out, "weights", f"{mname}_{fs}_seed0.weights.h5")
        if os.path.exists(wp):
            model.load_weights(wp)
        xs = as_list(x, names)
        one = [v[:1] for v in xs]
        bat = [v[:a.batch_size] for v in xs]
        f = tf.function(lambda *z: model(list(z), training=False))
        for _ in range(20):
            f(*one)
        ts = []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            r = f(*one)
            _ = r.numpy()
            ts.append((time.perf_counter() - t0) * 1e3)
        for _ in range(5):
            f(*bat)
        tb = []
        for _ in range(max(10, a.runs // 10)):
            t0 = time.perf_counter()
            _ = f(*bat).numpy()
            tb.append((time.perf_counter() - t0) * 1e3)
        rows.append({"model": mname, "device": device, "tf": tf.__version__,
                     "params": int(model.count_params()),
                     "latency_b1_median_ms": float(np.median(ts)), "latency_b1_p95_ms": float(np.percentile(ts, 95)),
                     f"latency_b{a.batch_size}_median_ms": float(np.median(tb)),
                     "per_sample_in_batch_ms": float(np.median(tb) / a.batch_size)})
        log(rows[-1])
        tf.keras.backend.clear_session()
    out = os.path.join(a.out, f"latency_{device}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    log("saved", out)


# --------------------------------------------------------------------------
# 8. OPTIONAL TUNING (equal budget for every model)
# --------------------------------------------------------------------------
def cmd_tune(a):
    try:
        import keras_tuner as kt
    except ImportError:
        raise SystemExit("pip install keras-tuner  (needed only for `tune`)")
    tf = setup_tf(0)
    ds = Dataset(a.work, W=a.window, max_users=a.max_users)
    xtr, ytr, _ = ds.windows("train", "full")
    xva, yva, _ = ds.windows("val", "full")
    os.makedirs(os.path.join(a.out, "tuning"), exist_ok=True)
    models = ALL_MODELS if a.models == ["all"] else a.models
    for mname in models:
        def hb(hp):
            cfg = {"lr": hp.Float("lr", 1e-4, 3e-3, sampling="log"),
                   "dropout": hp.Float("dropout", 0.0, 0.5, step=0.1),
                   "units": hp.Choice("units", [32, 64, 128]),
                   "l2": hp.Choice("l2", [0.0, 1e-5, 1e-4, 1e-3])}
            if mname not in ("DNN", "LSTM", "GRU"):
                cfg["d_model"] = hp.Choice("d_model", [32, 64])
            model, _ = build_model(mname, ds.W, xtr["num"].shape[2], True, ds.n_srv, ds.n_org,
                                   ds.n_app, ds.n_classes(), cfg)
            return model
        m0, names = build_model(mname, ds.W, xtr["num"].shape[2], True, ds.n_srv, ds.n_org, ds.n_app, ds.n_classes())
        tf.keras.backend.clear_session()
        tuner = kt.BayesianOptimization(hb, objective="val_loss", max_trials=a.trials, seed=0,
                                        directory=os.path.join(a.out, "tuning", "kt"), project_name=mname,
                                        overwrite=False)
        tuner.search(as_list(xtr, names), ytr, validation_data=(as_list(xva, names), yva),
                     epochs=a.epochs, batch_size=a.batch,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)], verbose=0)
        best = tuner.get_best_hyperparameters(1)[0].values
        trials = [{"trial": t.trial_id, **t.hyperparameters.values, "val_loss": t.score}
                  for t in tuner.oracle.trials.values()]
        json.dump({"best": best, "trials": trials, "budget": {"trials": a.trials, "epochs": a.epochs}},
                  open(os.path.join(a.out, "tuning", f"{mname}.json"), "w"), indent=2)
        log(mname, "best", best)
        tf.keras.backend.clear_session()


# --------------------------------------------------------------------------
# 9. REPORT
# --------------------------------------------------------------------------
def mean_ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], 0.0, 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.std(ddof=1), h


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return 1.0, b, c
    try:
        from scipy.stats import binomtest
        return float(binomtest(b, b + c, 0.5).pvalue), b, c
    except ImportError:                                   # scipy < 1.7
        from scipy.stats import binom_test
        return float(binom_test(b, b + c, 0.5)), b, c


def cmd_report(a):
    out = a.out
    keys = ["acc", "top3", "top5", "macro_f1", "weighted_f1", "balanced_acc", "acc_on_change",
            "acc_on_stay", "org_acc", "ece", "nll", "recall_head_>=1000", "recall_mid_100-999",
            "recall_tail_10-99", "recall_rare_1-9", "unseen_label_rate", "epochs", "train_time_s",
            "time_per_epoch_s", "params"]
    frames = []
    for f in ["metrics_baselines.csv", "metrics_models.csv"]:
        p = os.path.join(out, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("nothing to report yet")
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for (mname, fs), g in d.groupby(["model", "feature_set"], sort=False):
        r = {"model": mname, "feature_set": fs, "n_seeds": len(g)}
        for k in keys:
            if k in g:
                mu, sd, ci = mean_ci(g[k])
                r[k] = mu
                r[k + "_sd"] = sd
                r[k + "_ci95"] = ci
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    # McNemar vs persistence on the shared test set (seed 0 / first seed)
    pdir = os.path.join(out, "predictions")
    tests = []
    pp = os.path.join(pdir, "pred_Persistence_-_seed0.csv.gz")
    if os.path.exists(pp):
        base = pd.read_csv(pp)
        b_ok = (base.pred_server == base.label_server).values
        for f in sorted(os.listdir(pdir)):
            if f.startswith("pred_") and "_full_seed" in f:
                q = pd.read_csv(os.path.join(pdir, f))
                if len(q) != len(base):
                    continue
                ok = (q.pred_server == q.label_server).values
                pval, b, c = mcnemar(ok, b_ok)
                tests.append({"file": f, "acc": ok.mean(), "persistence_acc": b_ok.mean(),
                              "model_only_correct": b, "persistence_only_correct": c, "mcnemar_p": pval})
        pd.DataFrame(tests).to_csv(os.path.join(out, "mcnemar_vs_persistence.csv"), index=False)

    # Markdown + LaTeX main table (full feature set + baselines)
    main = summ[summ.feature_set.isin(["full", "-"])]

    def fmt(r, k, pct=True):
        if k not in r or pd.isna(r[k]):
            return "--"
        s = 100 if pct else 1
        if r.get("n_seeds", 1) > 1 and not pd.isna(r.get(k + "_sd", np.nan)) and r[k + "_sd"] > 0:
            return f"{r[k] * s:.2f} $\\pm$ {r[k + '_sd'] * s:.2f}"
        return f"{r[k] * s:.2f}"

    cols = [("acc", "Acc. (\\%)"), ("top5", "Top-5 (\\%)"), ("macro_f1", "Macro-F1 (\\%)"),
            ("weighted_f1", "Weighted-F1 (\\%)"), ("acc_on_change", "Acc. on changes (\\%)")]
    lines = ["\\begin{tabular}{|l|" + "c|" * len(cols) + "}", "\\hline",
             "\\textbf{Model} & " + " & ".join(f"\\textbf{{{c[1]}}}" for c in cols) + " \\\\ \\hline"]
    md = ["| Model | " + " | ".join(c[1].replace("\\%", "%") for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for _, r in main.iterrows():
        vals = [fmt(r, k) for k, _ in cols]
        lines.append(f"{r['model']} & " + " & ".join(vals) + " \\\\ \\hline")
        md.append(f"| {r['model']} | " + " | ".join(v.replace('$\\pm$', '±') for v in vals) + " |")
    lines.append("\\end{tabular}")
    open(os.path.join(out, "table_main.tex"), "w").write("\n".join(lines))
    abl = summ[summ.feature_set.isin(["full", "no_rtt", "no_srv"]) & ~summ.model.isin(["-"])]
    md += ["", "### Feature-set ablation (accuracy %, mean ± sd)", "", "| Model | full | no_rtt | no_srv |", "|---|---|---|---|"]
    for mname, g in abl.groupby("model", sort=False):
        cells = []
        for fs in ["full", "no_rtt", "no_srv"]:
            r = g[g.feature_set == fs]
            cells.append(fmt(r.iloc[0], "acc").replace("$\\pm$", "±") if len(r) else "--")
        md.append(f"| {mname} | " + " | ".join(cells) + " |")
    hp = os.path.join(out, "metrics_holdout.csv")
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        md += ["", "### Unseen users (5 user-disjoint folds, mean ± sd over folds)", "",
               "| Model | Acc. % | Acc. on changes % | Unseen-label rate % |", "|---|---|---|---|"]
        for (mname, fs), g in h.groupby(["model", "feature_set"], sort=False):
            mu, sd, _ = mean_ci(g.acc)
            mc, sc, _ = mean_ci(g.acc_on_change)
            mu2, _, _ = mean_ci(g.unseen_label_rate)
            md.append(f"| {mname} ({fs}) | {100 * mu:.2f} ± {100 * sd:.2f} | {100 * mc:.2f} ± {100 * sc:.2f} | {100 * mu2:.2f} |")
    for dev in ("GPU", "CPU"):
        lp = os.path.join(out, f"latency_{dev}.csv")
        if os.path.exists(lp):
            l = pd.read_csv(lp)
            md += ["", f"### Inference latency on {dev}", "", l.to_string(index=False)]
    if tests:
        md += ["", "### McNemar test vs. persistence (test set)", "",
               pd.DataFrame(tests).to_string(index=False)]
    open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
    log("wrote summary.csv, table_main.tex, REPORT.md in", out)
    print("\n".join(md))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, out=True):
        sp.add_argument("--work", default="work", help="folder with prepared data")
        if out:
            sp.add_argument("--out", default="results", help="results folder")
        sp.add_argument("--window", type=int, default=60)
        sp.add_argument("--max-users", type=int, default=None, help="debug: limit number of users")

    def trainargs(sp):
        sp.add_argument("--epochs", type=int, default=40)
        sp.add_argument("--batch", type=int, default=256)
        sp.add_argument("--patience", type=int, default=5)
        sp.add_argument("--verbose", action="store_true")
        sp.add_argument("--use-tuned", action="store_true", help="use results/tuning/<model>.json")

    sp = sub.add_parser("prepare")
    sp.add_argument("--data", required=True, help="GAViST5G CSV: CombinedALLGeo.csv (Canadian) or YouTubeCombinedGeo.csv (international, with --single-app YouTube)")
    sp.add_argument("--work", default="work")
    sp.add_argument("--single-app", default=None, help="e.g. YouTube for the international file")
    sp.add_argument("--app-map", default=None, help="CSV with columns User,app")
    sp.add_argument("--session-gap", type=int, default=300)

    sp = sub.add_parser("baselines")
    common(sp)

    sp = sub.add_parser("smoke")
    sp.add_argument("--work", default="work")
    sp.add_argument("--out", default="results")

    sp = sub.add_parser("train")
    common(sp)
    trainargs(sp)
    sp.add_argument("--models", nargs="+", default=["all"])
    sp.add_argument("--feature-sets", nargs="+", default=["full"], choices=list(FEATURE_SETS))
    sp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sp.add_argument("--save-all-preds", action="store_true")

    sp = sub.add_parser("holdout")
    common(sp)
    trainargs(sp)
    sp.add_argument("--models", nargs="*", default=["GRU"])
    sp.add_argument("--feature-sets", nargs="+", default=["full"], choices=list(FEATURE_SETS))
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("latency")
    common(sp)
    sp.add_argument("--models", nargs="+", default=["all"])
    sp.add_argument("--runs", type=int, default=300)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--use-tuned", action="store_true")

    sp = sub.add_parser("tune")
    common(sp)
    trainargs(sp)
    sp.add_argument("--models", nargs="+", default=["all"])
    sp.add_argument("--trials", type=int, default=15)

    sp = sub.add_parser("report")
    sp.add_argument("--out", default="results")

    a = p.parse_args()
    if a.cmd == "tune":
        a.epochs = min(a.epochs, 15)
    {"prepare": cmd_prepare, "baselines": cmd_baselines, "smoke": cmd_smoke, "train": cmd_train, "holdout": cmd_holdout,
     "latency": cmd_latency, "tune": cmd_tune, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
