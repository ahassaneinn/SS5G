#!/usr/bin/env python3
"""
allocate.py -- ns-3 stage 2 (called by run_ns3.sh; Python 3 standard library only).

Turns one decision set (ranked candidate servers from a model or a baseline) into the
flow schedule that eua_trace_sim.cc replays, applying an explicit, documented
allocation policy.

Control loop (epoch = 1 s, the granularity of the trace)
  For every active user-second (u,t), in increasing t, users in ascending id order:
    d_u      = estimated demand = user's rate in its previous active second (causal)
    L_s(t)   = sum of estimated demand already admitted to server s in epoch t
    C        = per-server capacity (--cap Mbps, 0 = unlimited)
  direct     s = top-1 candidate (no capacity check; overload shows up as queueing/loss in ns-3)
  lb         s = top-1 if L_s + d_u <= C, else the first server with spare capacity among
               A = [server used by u in its previous second (stickiness, same provider)]
                   + the K nearest servers of the SAME provider (AS/organisation) as the top-1,
               else (no spare capacity anywhere) the least-loaded server in [top-1] + A
               (ties -> earlier in the list, i.e. the model's choice is kept).
  lbk        as lb but A = [previous server] + model candidates 2..5 + K nearest same-provider
               servers ("model-ranked load balancing").
  leastload  load balancing only, no prediction: candidates = K nearest servers of the provider
               the user is currently using; first with spare capacity, else least loaded.
  The rule never splits a user-second across servers; reassignment happens at epoch boundaries
  (the old connection is left to drain, a new one is opened to the new server).

Outputs
  --out    flows CSV:  uid,app,t,server,delay_ms,dir,proto,bytes
  --stats  JSON with allocation-level metrics (accuracy after LB, override / switch rates,
           server overload and load-imbalance statistics computed from the true trace bytes).
"""
import argparse, csv, gzip, io, json, math, os, random, sys
from collections import defaultdict

GAMING = {"LOL", "TFT", "VAL"}
VIDEO = {"YouTube", "Netflix", "Prime", "Crave"}


def opn(path):
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def hav_km(a, b, c, d):
    a, b, c, d = map(math.radians, (a, b, c, d))
    h = math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def median(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="ns3_inputs")
    ap.add_argument("--decisions", required=True, help="e.g. GRU__s0")
    ap.add_argument("--policy", default="direct", choices=["direct", "lb", "lbk", "leastload"])
    ap.add_argument("--cap", type=float, default=0.0, help="per-server capacity in Mbps (0 = unlimited)")
    ap.add_argument("--scenario", default="real", choices=["real", "aligned"],
                    help="real: trace time of day; aligned: every user's test period starts at t=0 (high contention)")
    ap.add_argument("--density", type=int, default=1, help="replicate every user N times (scalability study)")
    ap.add_argument("--density-jitter", type=int, default=60, help="max random start offset (s) of replicas")
    ap.add_argument("--seed", type=int, default=1, help="seed for replica offsets only")
    ap.add_argument("--fallback-k", type=int, default=10)
    ap.add_argument("--stretch", type=float, default=1.5, help="route-length / great-circle ratio")
    ap.add_argument("--fiber-km-per-ms", type=float, default=200.0)
    ap.add_argument("--video-udp", default="tcp", choices=["tcp", "udp"],
                    help="video-app UDP (QUIC) bytes replayed over TCP (default) or plain UDP")
    ap.add_argument("--user-subset", type=int, default=0, help="smoke test: N users, stratified by app")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    C = a.cap * 1e6 if a.cap > 0 else math.inf

    # ---------------- load inputs
    srv = {}
    by_org = defaultdict(list)
    with opn(os.path.join(a.inputs, "servers.csv")) as f:
        for r in csv.DictReader(f):
            srv[r["ip"]] = (float(r["lat"]), float(r["lon"]), r["org"])
            by_org[r["org"]].append(r["ip"])
    users = {}
    with opn(os.path.join(a.inputs, "users.csv")) as f:
        for r in csv.DictReader(f):
            users[int(r["User"])] = (float(r["lat"]), float(r["lon"]), r["app"])
    gw = (median([u[0] for u in users.values()]), median([u[1] for u in users.values()]))

    keep_users = set(users)
    if a.user_subset > 0:
        per_app = defaultdict(list)
        for u in sorted(users):
            per_app[users[u][2]].append(u)
        keep_users, i = set(), 0
        while len(keep_users) < min(a.user_subset, len(users)):
            for app in sorted(per_app):
                if i < len(per_app[app]) and len(keep_users) < a.user_subset:
                    keep_users.add(per_app[app][i])
            i += 1

    steps = {}
    with opn(os.path.join(a.inputs, "steps.csv.gz")) as f:
        for r in csv.DictReader(f):
            u = int(r["User"])
            if u in keep_users:
                steps[(u, int(r["t"]))] = (r["app"], r["label"], r["last"], float(r["est_bps"]))
    traffic = defaultdict(list)
    with opn(os.path.join(a.inputs, "traffic.csv.gz")) as f:
        for r in csv.DictReader(f):
            k = (int(r["User"]), int(r["t"]))
            if k in steps:
                traffic[k].append((r["dir"], r["proto"], int(r["bytes"])))
    dec = {}
    path = os.path.join(a.inputs, "decisions", a.decisions + ".csv.gz")
    with opn(path) as f:
        for r in csv.DictReader(f):
            k = (int(r["User"]), int(r["t"]))
            if k in steps:
                dec[k] = [r[f"c{j}"] for j in range(1, 6) if r.get(f"c{j}")]
    missing = [k for k in steps if k not in dec]
    if missing:
        sys.exit(f"{len(missing)} decisions missing in {path}")

    # ---------------- helpers
    dist_cache = {}

    def srv_delay_ms(s):
        if s not in dist_cache:
            la, lo, _ = srv[s]
            dist_cache[s] = max(0.5, hav_km(gw[0], gw[1], la, lo) * a.stretch / a.fiber_km_per_ms)
        return dist_cache[s]

    near_cache = {}

    def nearest_same_org(s_ref):
        org = srv[s_ref][2]
        if org not in near_cache:
            lst = sorted(by_org[org], key=lambda x: (srv_delay_ms(x), x))
            near_cache[org] = lst[: max(a.fallback_k, 1) + 1]
        return near_cache[org]

    # ---------------- timeline (scenario, replicas)
    t_min_all = min(t for (_, t) in steps)
    t_min_user = defaultdict(lambda: 10 ** 12)
    for (u, t) in steps:
        t_min_user[u] = min(t_min_user[u], t)
    rng = random.Random(a.seed)
    events = []   # (t_sim, uid, (u, t))
    for u in sorted({u for (u, _) in steps}):
        for rep in range(a.density):
            off = 0 if rep == 0 else rng.randint(0, a.density_jitter)
            uid = u * 100 + rep
            for (uu, t) in [k for k in steps if k[0] == u]:
                base = (t - t_min_all) if a.scenario == "real" else (t - t_min_user[u])
                events.append((base + 1 + off, uid, (uu, t)))
    events.sort()

    # ---------------- allocation
    load = defaultdict(float)      # estimated load admitted in the current epoch
    cur_t = None
    prev_srv = {}                  # uid -> (t_sim, server)
    rows = []
    n = n_override = n_acc = n_pref_acc = 0
    true_load = defaultdict(float)  # (t_sim, server) -> actual bps
    users_on = defaultdict(list)    # (t_sim, server) -> uids
    assign_seq = defaultdict(list)  # uid -> [(t_sim, server, label)]

    for t_sim, uid, key in events:
        if t_sim != cur_t:
            load.clear()
            cur_t = t_sim
        app, label, last, est = steps[key]
        d = max(est, 1e3)
        cands = dec[key]
        pref = cands[0]
        prv = prev_srv.get(uid)
        prv_s = prv[1] if prv else None

        def fits(s):
            return load[s] + d <= C

        if a.policy == "direct" or (a.policy in ("lb", "lbk") and fits(pref)):
            s = pref
        elif a.policy in ("lb", "lbk"):
            alts = []
            if prv_s and prv_s != pref and srv[prv_s][2] == srv[pref][2]:
                alts.append(prv_s)
            if a.policy == "lbk":
                alts += cands[1:]
            alts += nearest_same_org(pref)
            seen, order = set(), []
            for x in [pref] + alts:
                if x not in seen:
                    seen.add(x)
                    order.append(x)
            s = next((x for x in order[1:] if fits(x)), None)
            if s is None:
                s = min(order, key=lambda x: (load[x] + d, order.index(x)))
        else:  # leastload: provider of the current server, nearest first, no prediction
            order = nearest_same_org(last)
            s = next((x for x in order if fits(x)), None)
            if s is None:
                s = min(order, key=lambda x: (load[x] + d, order.index(x)))

        load[s] += d
        prev_srv[uid] = (t_sim, s)
        n += 1
        n_override += (s != pref)
        n_acc += (s == label)
        n_pref_acc += (pref == label)
        assign_seq[uid].append((t_sim, s, label))
        dl = srv_delay_ms(s)
        tot = 0
        for (dr, proto, b) in traffic.get(key, []):
            if proto == "UDP" and app in VIDEO and a.video_udp == "tcp":
                proto = "TCP"
            rows.append((uid, app, t_sim, s, round(dl, 3), dr, proto, b))
            tot += b
        true_load[(t_sim, s)] += tot * 8.0
        users_on[(t_sim, s)].append(uid)

    # merge rows with the same key (UDP->TCP mapping can create duplicates)
    agg = defaultdict(int)
    meta = {}
    for (uid, app, t, s, dl, dr, proto, b) in rows:
        agg[(uid, t, s, dr, proto)] += b
        meta[(uid, t, s, dr, proto)] = (app, dl)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "app", "t", "server", "delay_ms", "dir", "proto", "bytes"])
        for k in sorted(agg):
            uid, t, s, dr, proto = k
            app, dl = meta[k]
            w.writerow([uid, app, t, s, dl, dr, proto, agg[k]])

    # ---------------- allocation-level statistics (true bytes)
    sw = sw_obs = pairs = 0
    for uid, seq in assign_seq.items():
        for (t0, s0, l0), (t1, s1, l1) in zip(seq, seq[1:]):
            if t1 - t0 <= 300:
                pairs += 1
                sw += (s1 != s0)
                sw_obs += (l1 != l0)
    over_us = over_b = tot_b = 0.0
    utils = []
    per_sec = defaultdict(list)
    for (t, s), bps in true_load.items():
        k = len(users_on[(t, s)])
        tot_b += bps
        per_sec[t].append(bps)
        if bps > C:
            over_us += k
            over_b += bps
        if C < math.inf:
            utils.append(bps / C)
    imb = [max(v) / (sum(v) / len(v)) for v in per_sec.values() if len(v) >= 2 and sum(v) > 0]
    srv_at = defaultdict(list)
    for (t, s) in true_load:
        srv_at[t].append(s)
    per_srv = defaultdict(float)
    for (t, s), bps in true_load.items():
        per_srv[s] += bps
    vals = list(per_srv.values())
    jain = (sum(vals) ** 2) / (len(vals) * sum(v * v for v in vals)) if vals else float("nan")
    utils.sort()

    def q(x, p):
        return x[min(len(x) - 1, int(p * (len(x) - 1) + 0.5))] if x else float("nan")

    st = {
        "decisions": a.decisions, "policy": a.policy, "cap_mbps": a.cap, "scenario": a.scenario,
        "density": a.density, "n_user_seconds": n, "n_users": len(assign_seq),
        "pref_acc": n_pref_acc / n, "assign_acc": n_acc / n, "lb_override_rate": n_override / n,
        "switch_rate": sw / max(pairs, 1), "observed_switch_rate": sw_obs / max(pairs, 1),
        "n_servers_used": len(per_srv),
        "overload_user_seconds_frac": over_us / n,
        "overload_bytes_frac": over_b / max(tot_b, 1),
        "util_p95": q(utils, 0.95), "util_max": utils[-1] if utils else float("nan"),
        "load_imbalance_max_over_mean": (sum(imb) / len(imb)) if imb else float("nan"),
        "seconds_with_2plus_servers": len(imb),
        "jain_server_bytes": jain,
        "max_active_servers_per_s": max(len(v) for v in per_sec.values()),
        "max_concurrent_users": max(sum(len(users_on[(t, x)]) for x in srv_at[t]) for t in srv_at),
        "sim_duration_s": max(t for (t, _) in true_load) + 1,
        "offered_GB": tot_b / 8e9,
        "flow_rows": len(agg),
    }
    with open(a.stats, "w") as f:
        json.dump(st, f, indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
