/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * eua_trace_sim_r2.cc  --  JCC revision 2: edge-server-level, trace-driven ns-3 evaluation
 *
 * Changes w.r.t. eua_trace_sim.cc (revision 1), all requested by Reviewer 1 in round 2:
 *   * Common random numbers: the per-user access delay and every intra-second send offset are
 *     drawn from a hash of (run, user, second, direction, protocol, piece), so for one --run value
 *     every allocator sees identical random timing (paired comparisons across allocators).
 *   * Server load measured directly: bits sent on each server link in every 1-s epoch -> variance of
 *     server utilisation across the servers in use, Jain's index, utilisation percentiles, and
 *     overloaded server-epochs / user-seconds from actual queue drops (--outSrv).
 *   * Residual traffic measured at every epoch boundary: bytes still queued on each server link plus
 *     bytes not yet sent or acknowledged by the senders of that server's flows.  With --checks (the
 *     capacity checks logged by allocate.py) it counts false admissions: checks that passed although
 *     the measured residual left no spare capacity.
 *   * Reassignments measured: users whose server changes between consecutive active seconds, and how
 *     many of them still had traffic draining on the old server's connection.
 *
 * Replaces the 40 *_sim_*.cc scripts of the submitted version with ONE program.
 * The allocation (which server serves which user in every second, incl. load balancing)
 * is computed beforehand by allocate.py; this program replays that schedule packet by packet.
 *
 * Topology (star, core/edge level -- NOT a 5G radio model; see README "Scope"):
 *
 *     user_i --[access link: accessRate, one-way delay ~U(accessMinMs, accessMaxMs) per user]--+
 *                                                                                            GW
 *     server_j --[server link: capacity = --cap Mbps, delay = great-circle(GW,server)*stretch/c_fiber]--+
 *
 *   * The per-server capacity is a real bottleneck (link rate + finite queue), not a start-time
 *     delay as in the submitted scripts (audit item N2).
 *   * Propagation delay comes from geographic distance in km (audit item N3: the old scripts
 *     treated degrees as metres, giving ~0 ms).
 *   * Traffic = the dominant-server traffic of each TEST-period user-second of the trace
 *     (identical bytes for every allocator, audit item N1), sent in the second it was observed.
 *     TCP: the second's bytes are pushed in --subChunks pieces at random offsets inside the second
 *     (application-limited TCP; backlog carries over).  UDP: paced packets with random gaps.
 *   * Randomness (--run): per-user access delay, intra-second timing. Use several --run values
 *     per configuration (audit item N5, Reviewer 1 #14).
 *
 * Metrics (per group = ALL / VIDEO / GAMING / each app, per direction = ALL / DL / UL):
 *   offered / delivered bytes, delivery ratio, goodput per active user-second, legacy per-flow
 *   throughput, packet loss, one-way delay mean / p50 / p95 / p99 (FlowMonitor histograms),
 *   jitter, share of packets under 50 / 100 ms, and for video downlink a stall proxy:
 *   share of offered seconds whose bytes are not fully delivered within --stallThr seconds.
 *
 * Build: copy to <ns-3>/scratch/ and run "./ns3 build scratch/eua_trace_sim_r2"  (ns-3.36 or newer)
 * Run  : ./ns3 run "eua_trace_sim_r2 --flows=flows.csv --cap=10 --run=1 --out=res.csv --outSrv=srv.csv"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("EuaTraceSimR2");

// ------------------------------------------------------------------------------------------
// Common random numbers: a uniform number in [0,1) from a hash of its identifiers
// ------------------------------------------------------------------------------------------
static uint64_t
Mix64 (uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static double
HashU01 (uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  uint64_t h = Mix64 (a);
  h = Mix64 (h ^ b);
  h = Mix64 (h ^ c);
  h = Mix64 (h ^ d);
  return (double) (h >> 11) * (1.0 / 9007199254740992.0);   // 53-bit mantissa
}

// ------------------------------------------------------------------------------------------
// Input rows and flow keys
// ------------------------------------------------------------------------------------------
struct Row
{
  uint32_t uid;
  std::string app;
  int64_t t;
  std::string server;
  double delayMs;
  bool dl;
  bool tcp;
  uint64_t bytes;
};

struct FlowKey
{
  uint32_t uid;
  std::string app;
  std::string server;
  bool dl;
  bool tcp;
  uint16_t port;
  std::vector<std::pair<int64_t, uint64_t>> sched; // (second, bytes)
  uint64_t offered = 0;
  uint64_t rxApp = 0;                       // bytes received by the sink
  std::map<int64_t, uint64_t> rxPerSec;     // bytes received per simulation second
  uint32_t srvI = 0;                        // server index
  void *sender = nullptr;                   // TraceSender* (for residual measurement)
};

static std::vector<FlowKey> g_keys;

static const std::set<std::string> kGaming = {"LOL", "TFT", "VAL"};
static const std::set<std::string> kVideo = {"YouTube", "Netflix", "Prime", "Crave"};

static void
SinkRx (uint32_t key, Ptr<const Packet> p, const Address &from)
{
  (void) from;
  FlowKey &k = g_keys[key];
  k.rxApp += p->GetSize ();
  k.rxPerSec[(int64_t) std::floor (Simulator::Now ().GetSeconds ())] += p->GetSize ();
}

// ------------------------------------------------------------------------------------------
// TraceSender: replays a per-second byte schedule over TCP or UDP
// ------------------------------------------------------------------------------------------
class TraceSender : public Application
{
public:
  static TypeId
  GetTypeId ()
  {
    static TypeId tid = TypeId ("ns3::EuaTraceSender")
                            .SetParent<Application> ()
                            .SetGroupName ("Applications")
                            .AddConstructor<TraceSender> ();
    return tid;
  }

  TraceSender () = default;
  ~TraceSender () override = default;

  void
  Configure (Address peer, bool tcp, uint32_t udpPayload, uint32_t subChunks, uint32_t writeSize,
             const std::vector<std::pair<int64_t, uint64_t>> *sched, uint64_t run, uint64_t flowId,
             uint32_t txBuf)
  {
    m_peer = peer;
    m_tcp = tcp;
    m_udpPayload = udpPayload;
    m_subChunks = std::max<uint32_t> (1, subChunks);
    m_writeSize = writeSize;
    m_sched = sched;
    m_run = run;
    m_flowId = flowId;
    m_txBuf = txBuf;
  }

  uint64_t Pending () const { return m_pending; }

  // bytes of this flow not yet delivered to the network: application backlog plus the TCP send
  // buffer (unsent and unacknowledged bytes)
  uint64_t
  Residual () const
  {
    uint64_t r = m_pending;
    if (m_tcp && m_socket)
      {
        uint32_t avail = m_socket->GetTxAvailable ();
        if (avail < m_txBuf)
          r += (uint64_t) (m_txBuf - avail);
      }
    return r;
  }

private:
  void
  StartApplication () override
  {
    TypeId tid = m_tcp ? TcpSocketFactory::GetTypeId () : UdpSocketFactory::GetTypeId ();
    m_socket = Socket::CreateSocket (GetNode (), tid);
    m_socket->Bind ();
    if (m_tcp)
      {
        m_socket->SetConnectCallback (MakeCallback (&TraceSender::Connected, this),
                                      MakeCallback (&TraceSender::ConnectFailed, this));
        m_socket->SetSendCallback (MakeCallback (&TraceSender::TrySend, this));
      }
    m_socket->Connect (m_peer);
    if (!m_tcp)
      {
        m_connected = true;
      }
    double now = Simulator::Now ().GetSeconds ();
    for (const auto &e : *m_sched)
      {
        double t0 = std::max (0.0, (double) e.first - now);
        if (m_tcp)
          {
            uint64_t per = e.second / m_subChunks;
            uint64_t rem = e.second - per * m_subChunks;
            for (uint32_t j = 0; j < m_subChunks; ++j)
              {
                uint64_t b = per + (j == m_subChunks - 1 ? rem : 0);
                if (b == 0)
                  continue;
                double off = (j + HashU01 (m_run, m_flowId, (uint64_t) e.first, j)) / m_subChunks;
                m_events.push_back (
                    Simulator::Schedule (Seconds (t0 + off), &TraceSender::TcpChunk, this, b));
              }
          }
        else
          {
            uint32_t n = (uint32_t) ((e.second + m_udpPayload - 1) / m_udpPayload);
            if (n == 0)
              continue;
            uint32_t last = (uint32_t) (e.second - (uint64_t) (n - 1) * m_udpPayload);
            double gap = 1.0 / n;
            double off = gap * HashU01 (m_run, m_flowId, (uint64_t) e.first, 0xFFFFFFFFULL);
            m_events.push_back (Simulator::Schedule (Seconds (t0 + off), &TraceSender::UdpBurst,
                                                     this, n, last, gap, (uint64_t) e.first));
          }
      }
  }

  void
  StopApplication () override
  {
    for (auto &ev : m_events)
      {
        Simulator::Cancel (ev);
      }
  }

  void
  TcpChunk (uint64_t bytes)
  {
    m_pending += bytes;
    TrySend (m_socket, 0);
  }

  void
  Connected (Ptr<Socket> s)
  {
    (void) s;
    m_connected = true;
    TrySend (m_socket, 0);
  }

  void
  ConnectFailed (Ptr<Socket> s)
  {
    (void) s;
    NS_LOG_WARN ("TCP connection failed at " << Simulator::Now ().GetSeconds ());
  }

  void
  TrySend (Ptr<Socket> s, uint32_t avail)
  {
    (void) s;
    (void) avail;
    if (!m_connected)
      return;
    while (m_pending > 0)
      {
        uint32_t space = m_socket->GetTxAvailable ();
        if (space == 0)
          break;
        uint64_t n = std::min<uint64_t> (m_pending, std::min<uint64_t> (space, m_writeSize));
        int r = m_socket->Send (Create<Packet> ((uint32_t) n));
        if (r <= 0)
          break;
        m_pending -= (uint64_t) r;
      }
  }

  void
  UdpBurst (uint32_t left, uint32_t lastSize, double gap, uint64_t sec)
  {
    uint32_t sz = (left == 1) ? lastSize : m_udpPayload;
    m_socket->Send (Create<Packet> (sz));
    if (left > 1)
      {
        double g = gap * (0.5 + HashU01 (m_run, m_flowId, sec, left)); // mean gap preserved
        Simulator::Schedule (Seconds (g), &TraceSender::UdpBurst, this, left - 1, lastSize, gap, sec);
      }
  }

  Ptr<Socket> m_socket;
  Address m_peer;
  bool m_tcp = true;
  bool m_connected = false;
  uint32_t m_udpPayload = 1200;
  uint32_t m_subChunks = 10;
  uint32_t m_writeSize = 65536;
  uint64_t m_pending = 0;
  const std::vector<std::pair<int64_t, uint64_t>> *m_sched = nullptr;
  uint64_t m_run = 1;
  uint64_t m_flowId = 0;
  uint32_t m_txBuf = 4 * 1024 * 1024;
  std::vector<EventId> m_events;
};

NS_OBJECT_ENSURE_REGISTERED (TraceSender);

// ------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------
static std::chrono::steady_clock::time_point g_wall0;

static void
Progress (int i)
{
  double el = std::chrono::duration<double> (std::chrono::steady_clock::now () - g_wall0).count ();
  std::cout << "  progress " << i * 10 << "%  wall " << std::fixed << std::setprecision (0) << el
            << " s" << std::endl;
}
static std::vector<std::string>
SplitCsv (const std::string &line)
{
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss (line);
  while (std::getline (ss, cur, ','))
    out.push_back (cur);
  return out;
}

static uint32_t
QueuePackets (double rateBps, double bufferMs, uint32_t minP, uint32_t maxP)
{
  double p = rateBps * bufferMs / 1000.0 / (1500.0 * 8.0);
  return (uint32_t) std::max<double> (minP, std::min<double> (maxP, std::ceil (p)));
}

static Ipv4InterfaceContainer
MakeLink (Ptr<Node> a, Ptr<Node> b, double rateBps, double delayMs, const std::string &qdisc,
          uint32_t qPkts, Ipv4AddressHelper &addr, NetDeviceContainer *devOut = nullptr,
          QueueDiscContainer *qdOut = nullptr)
{
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ((uint64_t) rateBps)));
  p2p.SetChannelAttribute ("Delay", TimeValue (Seconds (delayMs / 1000.0)));
  p2p.SetQueue ("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue (QueueSize ("1p")));
  NetDeviceContainer d = p2p.Install (a, b);

  TrafficControlHelper tch;
  QueueSizeValue qs (QueueSize (QueueSizeUnit::PACKETS, qPkts));
  if (qdisc == "codel")
    tch.SetRootQueueDisc ("ns3::CoDelQueueDisc", "MaxSize", qs);
  else if (qdisc == "fqcodel")
    tch.SetRootQueueDisc ("ns3::FqCoDelQueueDisc", "MaxSize", qs);
  else
    tch.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", qs);
  QueueDiscContainer qd = tch.Install (d);
  if (devOut)
    *devOut = d;
  if (qdOut)
    *qdOut = qd;

  Ipv4InterfaceContainer ifc = addr.Assign (d);
  addr.NewNetwork ();
  return ifc;
}

struct Acc
{
  uint32_t nFlows = 0;
  uint64_t txPkts = 0, rxPkts = 0, txBytes = 0, rxBytes = 0;
  double delaySum = 0, jitterSum = 0;
  uint64_t jitterN = 0;
  double thrSum = 0;
  uint32_t thrN = 0;
  std::map<int64_t, uint64_t> hist; // delay histogram, bin index -> count
  uint64_t offered = 0, rxApp = 0, activeUserSec = 0;
  uint64_t stallN = 0, stallDen = 0;
  double lagSum = 0;
};

static double
HistQuantile (const std::map<int64_t, uint64_t> &h, double q, double w)
{
  uint64_t tot = 0;
  for (const auto &kv : h)
    tot += kv.second;
  if (tot == 0)
    return NAN;
  double target = q * (double) tot;
  uint64_t cum = 0;
  for (const auto &kv : h)
    {
      cum += kv.second;
      if ((double) cum >= target)
        return (kv.first + 0.5) * w;
    }
  return (h.rbegin ()->first + 0.5) * w;
}

static double
HistShareBelow (const std::map<int64_t, uint64_t> &h, double thr, double w)
{
  uint64_t tot = 0, below = 0;
  for (const auto &kv : h)
    {
      tot += kv.second;
      if ((kv.first + 1) * w <= thr + 1e-12)
        below += kv.second;
    }
  return tot ? (double) below / (double) tot : NAN;
}

static std::vector<std::string>
GroupsOf (const std::string &app)
{
  std::vector<std::string> g = {"ALL", app};
  if (kVideo.count (app))
    g.push_back ("VIDEO");
  if (kGaming.count (app))
    g.push_back ("GAMING");
  return g;
}

// ------------------------------------------------------------------------------------------
// Server-level measurement (revision 2)
// ------------------------------------------------------------------------------------------
struct SrvMeas
{
  Ptr<QueueDisc> qd[2];             // [0] server side (DL), [1] gateway side (UL)
  Ptr<Queue<Packet>> dq[2];         // device queues
  uint64_t txBits = 0;              // bits sent on the link in the current epoch (both directions)
  uint64_t dropBytes = 0;           // bytes dropped by the queue discs in the current epoch
  double totBits = 0;               // whole run
  uint64_t totDropBytes = 0;
};
static std::vector<SrvMeas> g_srv;

static void
SrvTx (uint32_t j, Ptr<const Packet> p)
{
  g_srv[j].txBits += 8ULL * p->GetSize ();
}

static void
SrvDrop (uint32_t j, Ptr<const QueueDiscItem> item)
{
  g_srv[j].dropBytes += item->GetSize ();
}

struct Check
{
  uint32_t srv;      // index, or UINT32_MAX if the server carries no traffic in this run
  double loadBps, dBps, residEstBits;
  bool passed;
};

struct EpochState
{
  double capBps = 0;                 // per-server capacity (bit/s), 0 = unlimited
  double linkBps = 0;                // link rate used for utilisation
  int64_t tFirst = 0, tEnd = 0;
  // measured load
  uint64_t nActiveEpochs = 0, nEpochs2plus = 0;
  double sumVar = 0, sumVar2 = 0;
  std::vector<double> utils;         // utilisation of every active server-epoch
  uint64_t nActiveSrvEpochs = 0, nOverSrvEpochs = 0, overUserSec = 0, userSec = 0;
  // residual
  std::vector<double> residAssigned; // residual (in epochs of capacity) at servers in use
  // checks
  std::unordered_map<int64_t, std::vector<Check>> checks;
  uint64_t nChecks = 0, nPassed = 0, nFalseAdm = 0, nFailed = 0, nFalseRej = 0;
  double sumEstErr = 0;
  // reassignments
  std::unordered_map<int64_t, std::vector<std::pair<uint32_t, uint32_t>>> reassign; // t -> (uid, oldSrv)
  uint64_t nReassign = 0, nReassignDrain = 0, nPairs = 0;
  double drainBytes = 0;
  // schedule lookups
  std::unordered_map<int64_t, std::vector<std::pair<uint32_t, uint32_t>>> usersOn; // t -> (srv, nUsers)
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> keysOf;         // (uid, srv) -> keys
  // open flow keys for the residual measurement
  std::vector<uint32_t> keyOrder;    // keys sorted by first second
  size_t nextKey = 0;
  std::vector<uint32_t> openKeys;
  std::vector<uint64_t> keyRes;      // residual per key at the current tick
  std::vector<int64_t> keyFirst, keyLast;
};
static EpochState g_ep;

static uint64_t
KeyResidual (uint32_t ki)
{
  const FlowKey &fk = g_keys[ki];
  if (!fk.tcp || !fk.sender)
    return 0;
  return static_cast<const TraceSender *> (fk.sender)->Residual ();
}

static void
EpochTick (int64_t T)
{
  EpochState &e = g_ep;
  // ---- (1) close epoch T-1: measured load per server
  if (T - 1 >= e.tFirst)
    {
      std::vector<double> u;
      std::unordered_set<uint32_t> over;
      for (uint32_t j = 0; j < g_srv.size (); ++j)
        {
          SrvMeas &m = g_srv[j];
          if (m.txBits > 0)
            {
              double ut = (double) m.txBits / e.linkBps;
              u.push_back (ut);
              e.utils.push_back (ut);
              e.nActiveSrvEpochs++;
              if (m.dropBytes > 0)
                {
                  e.nOverSrvEpochs++;
                  over.insert (j);
                }
            }
          m.totBits += (double) m.txBits;
          m.totDropBytes += m.dropBytes;
          m.txBits = 0;
          m.dropBytes = 0;
        }
      if (!u.empty ())
        {
          double mu = 0, v = 0;
          for (double x : u)
            mu += x;
          mu /= u.size ();
          for (double x : u)
            v += (x - mu) * (x - mu);
          v /= u.size ();
          e.nActiveEpochs++;
          e.sumVar += v;
          if (u.size () >= 2)
            {
              e.nEpochs2plus++;
              e.sumVar2 += v;
            }
        }
      auto it = e.usersOn.find (T - 1);
      if (it != e.usersOn.end ())
        for (const auto &su : it->second)
          {
            e.userSec += su.second;
            if (over.count (su.first))
              e.overUserSec += su.second;
          }
    }

  // ---- (2) residual traffic at the start of epoch T
  while (e.nextKey < e.keyOrder.size () && e.keyFirst[e.keyOrder[e.nextKey]] <= T)
    e.openKeys.push_back (e.keyOrder[e.nextKey++]);
  std::vector<double> resBits (g_srv.size (), 0.0);
  for (size_t i = 0; i < e.openKeys.size ();)
    {
      uint32_t ki = e.openKeys[i];
      uint64_t r = KeyResidual (ki);
      e.keyRes[ki] = r;
      resBits[g_keys[ki].srvI] += 8.0 * (double) r;
      if (r == 0 && e.keyLast[ki] < T)
        {
          e.openKeys[i] = e.openKeys.back ();
          e.openKeys.pop_back ();
        }
      else
        ++i;
    }
  for (uint32_t j = 0; j < g_srv.size (); ++j)
    for (int s = 0; s < 2; ++s)
      {
        if (g_srv[j].qd[s])
          resBits[j] += 8.0 * (double) g_srv[j].qd[s]->GetNBytes ();
        if (g_srv[j].dq[s])
          resBits[j] += 8.0 * (double) g_srv[j].dq[s]->GetNBytes ();
      }
  double capEpochBits = (e.capBps > 0 ? e.capBps : e.linkBps) * 1.0;
  auto uit = e.usersOn.find (T);
  if (uit != e.usersOn.end ())
    for (const auto &su : uit->second)
      e.residAssigned.push_back (resBits[su.first] / capEpochBits);

  // capacity checks of epoch T (from allocate.py) against the measured residual
  auto cit = e.checks.find (T);
  if (cit != e.checks.end () && e.capBps > 0)
    for (const Check &c : cit->second)
      {
        double r = (c.srv == UINT32_MAX) ? 0.0 : resBits[c.srv];
        bool fitsMeasured = c.loadBps + c.dBps + r <= e.capBps;
        e.nChecks++;
        e.sumEstErr += std::fabs (c.residEstBits - r) / capEpochBits;
        if (c.passed)
          {
            e.nPassed++;
            if (!fitsMeasured)
              e.nFalseAdm++;
          }
        else
          {
            e.nFailed++;
            if (fitsMeasured)
              e.nFalseRej++;
          }
      }

  // reassignments at epoch T: is the old connection still draining?
  auto rit = e.reassign.find (T);
  if (rit != e.reassign.end ())
    for (const auto &ro : rit->second)
      {
        e.nReassign++;
        uint64_t b = 0;
        auto kit = e.keysOf.find ({ro.first, ro.second});
        if (kit != e.keysOf.end ())
          for (uint32_t ki : kit->second)
            {
              if (e.keyFirst[ki] <= T)
                b += KeyResidual (ki);
            }
        if (b > 0)
          {
            e.nReassignDrain++;
            e.drainBytes += (double) b;
          }
      }

  if (T + 1 <= e.tEnd)
    Simulator::Schedule (Seconds (1.0), &EpochTick, T + 1);
}

static double
Quantile (std::vector<double> v, double q)
{
  if (v.empty ())
    return NAN;
  std::sort (v.begin (), v.end ());
  size_t i = (size_t) std::min<double> (v.size () - 1, std::floor (q * (v.size () - 1) + 0.5));
  return v[i];
}

// ------------------------------------------------------------------------------------------
int
main (int argc, char *argv[])
{
  auto wall0 = std::chrono::steady_clock::now ();

  std::string flowsFile = "flows.csv";
  std::string outFile = "eua_result.csv";
  std::string outSrv = "";          // server-level metrics (default: <out without .csv>_srv.csv)
  std::string checksFile = "";      // capacity checks written by allocate.py --checks
  double capMbps = 0.0;            // 0 = unlimited
  double unlimitedMbps = 10000.0;  // link rate used when cap = 0
  double accessMbps = 1000.0;
  double accessMinMs = 5.0, accessMaxMs = 10.0;
  double bufferMs = 100.0;
  uint32_t minQ = 10, maxQ = 10000;
  std::string qdisc = "fifo";
  std::string tcpType = "ns3::TcpCubic";
  uint32_t seed = 1, run = 1;
  uint32_t udpPayload = 1200, gamingPayload = 200;
  uint32_t subChunks = 10;
  uint32_t bufBytes = 4 * 1024 * 1024;
  double drain = 30.0;
  double stallThr = 2.0;
  double binW = 0.0005;
  bool verbose = true;

  CommandLine cmd;
  cmd.AddValue ("flows", "flow schedule from allocate.py", flowsFile);
  cmd.AddValue ("out", "output CSV (one row per group x direction)", outFile);
  cmd.AddValue ("cap", "per-server capacity in Mbps (0 = unlimited)", capMbps);
  cmd.AddValue ("unlimitedMbps", "server link rate when cap = 0", unlimitedMbps);
  cmd.AddValue ("accessMbps", "user access link rate (Mbps)", accessMbps);
  cmd.AddValue ("accessMinMs", "min one-way access delay (ms)", accessMinMs);
  cmd.AddValue ("accessMaxMs", "max one-way access delay (ms)", accessMaxMs);
  cmd.AddValue ("bufferMs", "queue size expressed in ms at link rate", bufferMs);
  cmd.AddValue ("minQ", "min queue size (packets)", minQ);
  cmd.AddValue ("maxQ", "max queue size (packets)", maxQ);
  cmd.AddValue ("qdisc", "queue discipline on every link: fifo | codel | fqcodel", qdisc);
  cmd.AddValue ("tcp", "TCP variant TypeId", tcpType);
  cmd.AddValue ("seed", "RNG seed", seed);
  cmd.AddValue ("run", "RNG run number (independent replication)", run);
  cmd.AddValue ("udpPayload", "UDP payload bytes (video / other)", udpPayload);
  cmd.AddValue ("gamingPayload", "UDP payload bytes for gaming apps", gamingPayload);
  cmd.AddValue ("subChunks", "TCP: pieces per second", subChunks);
  cmd.AddValue ("tcpBuf", "TCP send/receive buffer (bytes)", bufBytes);
  cmd.AddValue ("drain", "seconds simulated after the last scheduled second", drain);
  cmd.AddValue ("stallThr", "stall proxy threshold (s)", stallThr);
  cmd.AddValue ("binW", "delay histogram bin width (s)", binW);
  cmd.AddValue ("verbose", "print progress", verbose);
  cmd.AddValue ("outSrv", "server-level metrics CSV (load variance, residual, reassignments)", outSrv);
  cmd.AddValue ("checks", "capacity checks from allocate.py --checks (optional)", checksFile);
  cmd.Parse (argc, argv);
  if (outSrv.empty ())
    {
      outSrv = outFile;
      if (outSrv.size () > 4 && outSrv.substr (outSrv.size () - 4) == ".csv")
        outSrv = outSrv.substr (0, outSrv.size () - 4);
      else if (outSrv.size () > 9 && outSrv.substr (outSrv.size () - 9) == ".csv.part")
        outSrv = outSrv.substr (0, outSrv.size () - 9);
      outSrv += "_srv.csv";
    }

  RngSeedManager::SetSeed (seed);
  RngSeedManager::SetRun (run);

  TypeId tcpTid;
  if (!TypeId::LookupByNameFailSafe (tcpType, &tcpTid))
    {
      std::cerr << "TCP type " << tcpType << " not found, using ns3::TcpNewReno\n";
      tcpTid = TypeId::LookupByName ("ns3::TcpNewReno");
    }
  Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (tcpTid));
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (bufBytes));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (bufBytes));

  // ---------------- read the flow schedule
  std::ifstream in (flowsFile);
  if (!in)
    {
      std::cerr << "cannot open " << flowsFile << "\n";
      return 1;
    }
  std::string line;
  std::getline (in, line); // header: uid,app,t,server,delay_ms,dir,proto,bytes
  std::vector<Row> rows;
  while (std::getline (in, line))
    {
      if (!line.empty () && line.back () == '\r')
        line.pop_back ();
      auto tk = SplitCsv (line);
      if (tk.size () != 8)
        continue;
      Row r;
      r.uid = (uint32_t) std::stoul (tk[0]);
      r.app = tk[1];
      r.t = std::stoll (tk[2]);
      r.server = tk[3];
      r.delayMs = std::stod (tk[4]);
      r.dl = (tk[5] == "DL");
      r.tcp = (tk[6] == "TCP");
      r.bytes = std::stoull (tk[7]);
      if (r.bytes > 0)
        rows.push_back (r);
    }
  if (rows.empty ())
    {
      std::cerr << "no rows in " << flowsFile << "\n";
      return 1;
    }

  // ---------------- nodes
  std::map<uint32_t, uint32_t> userIdx;     // uid -> index
  std::map<std::string, uint32_t> srvIdx;   // server ip -> index
  std::map<std::string, double> srvDelay;
  for (const auto &r : rows)
    {
      if (!userIdx.count (r.uid))
        {
          uint32_t i = (uint32_t) userIdx.size ();
          userIdx[r.uid] = i;
        }
      if (!srvIdx.count (r.server))
        {
          uint32_t i = (uint32_t) srvIdx.size ();
          srvIdx[r.server] = i;
          srvDelay[r.server] = r.delayMs;
        }
    }
  NodeContainer gw, users, servers;
  gw.Create (1);
  users.Create ((uint32_t) userIdx.size ());
  servers.Create ((uint32_t) srvIdx.size ());
  InternetStackHelper internet;
  internet.Install (gw);
  internet.Install (users);
  internet.Install (servers);

  Ipv4AddressHelper addr;
  addr.SetBase ("10.0.0.0", "255.255.255.252");

  std::vector<Ipv4Address> userIp (users.GetN ()), srvIp (servers.GetN ());
  double accessBps = accessMbps * 1e6;
  for (const auto &kv : userIdx) // ascending uid -> same delays for every allocator (same run)
    {
      // common random numbers: the same delay for this user and run under every allocator
      double dMs = accessMinMs + (accessMaxMs - accessMinMs) * HashU01 (run, kv.first, 0xACCE55ULL, 0);
      auto ifc = MakeLink (users.Get (kv.second), gw.Get (0), accessBps, dMs, qdisc,
                           QueuePackets (accessBps, bufferMs, minQ, maxQ), addr);
      userIp[kv.second] = ifc.GetAddress (0);
    }
  double srvBps = (capMbps > 0 ? capMbps : unlimitedMbps) * 1e6;
  g_srv.assign (servers.GetN (), SrvMeas ());
  for (const auto &kv : srvIdx)
    {
      NetDeviceContainer dv;
      QueueDiscContainer qd;
      auto ifc = MakeLink (servers.Get (kv.second), gw.Get (0), srvBps, srvDelay[kv.first], qdisc,
                           QueuePackets (srvBps, bufferMs, minQ, maxQ), addr, &dv, &qd);
      srvIp[kv.second] = ifc.GetAddress (0);
      uint32_t j = kv.second;
      for (uint32_t s = 0; s < 2; ++s)
        {
          Ptr<PointToPointNetDevice> pd = DynamicCast<PointToPointNetDevice> (dv.Get (s));
          pd->TraceConnectWithoutContext ("PhyTxEnd", MakeBoundCallback (&SrvTx, j));
          g_srv[j].dq[s] = pd->GetQueue ();
          g_srv[j].qd[s] = qd.Get (s);
          qd.Get (s)->TraceConnectWithoutContext ("Drop", MakeBoundCallback (&SrvDrop, j));
        }
    }
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // ---------------- flow keys: (uid, server, dir, proto)
  std::map<std::tuple<uint32_t, std::string, bool, bool>, uint32_t> keyOf;
  int64_t lastT = 0;
  // per group/dir offered bytes and active user-seconds
  std::map<std::pair<std::string, std::string>, std::set<std::pair<uint32_t, int64_t>>> activeSec;
  std::map<uint32_t, std::map<int64_t, uint64_t>> dlOfferedPerUser; // video DL, for stall proxy
  std::map<uint32_t, std::string> appOfUser;
  for (const auto &r : rows)
    {
      auto k = std::make_tuple (r.uid, r.server, r.dl, r.tcp);
      auto it = keyOf.find (k);
      uint32_t ki;
      if (it == keyOf.end ())
        {
          ki = (uint32_t) g_keys.size ();
          keyOf[k] = ki;
          FlowKey fk;
          fk.uid = r.uid;
          fk.app = r.app;
          fk.server = r.server;
          fk.dl = r.dl;
          fk.tcp = r.tcp;
          fk.port = 0;
          g_keys.push_back (fk);
        }
      else
        ki = it->second;
      g_keys[ki].sched.emplace_back (r.t, r.bytes);
      g_keys[ki].offered += r.bytes;
      lastT = std::max (lastT, r.t);
      appOfUser[r.uid] = r.app;
      for (const auto &g : GroupsOf (r.app))
        {
          activeSec[{g, "ALL"}].insert ({r.uid, r.t});
          activeSec[{g, r.dl ? "DL" : "UL"}].insert ({r.uid, r.t});
        }
      if (r.dl && kVideo.count (r.app))
        dlOfferedPerUser[r.uid][r.t] += r.bytes;
    }
  if (g_keys.size () > 39000) // ports 10000.. must stay below the ephemeral range (49152)
    {
      std::cerr << "too many flow keys for the port space: " << g_keys.size () << "\n";
      return 1;
    }

  // ---------------- applications
  std::map<uint16_t, uint32_t> portToKey;
  uint16_t nextPort = 10000;
  for (uint32_t ki = 0; ki < g_keys.size (); ++ki)
    {
      FlowKey &fk = g_keys[ki];
      std::sort (fk.sched.begin (), fk.sched.end ());
      fk.port = nextPort++;
      portToKey[fk.port] = ki;
      Ptr<Node> un = users.Get (userIdx[fk.uid]);
      Ptr<Node> sn = servers.Get (srvIdx[fk.server]);
      Ptr<Node> sender = fk.dl ? sn : un;
      Ptr<Node> receiver = fk.dl ? un : sn;
      Ipv4Address rxIp = fk.dl ? userIp[userIdx[fk.uid]] : srvIp[srvIdx[fk.server]];

      PacketSinkHelper sinkH (fk.tcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory",
                              InetSocketAddress (Ipv4Address::GetAny (), fk.port));
      ApplicationContainer sa = sinkH.Install (receiver);
      sa.Start (Seconds (0.0));
      Ptr<PacketSink> sink = DynamicCast<PacketSink> (sa.Get (0));
      sink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&SinkRx, ki));

      Ptr<TraceSender> app = CreateObject<TraceSender> ();
      uint32_t payload = kGaming.count (fk.app) ? gamingPayload : udpPayload;
      // flow identity for common random numbers: user, direction, protocol (not the server)
      uint64_t flowId = ((uint64_t) fk.uid << 2) | ((uint64_t) fk.dl << 1) | (uint64_t) fk.tcp;
      app->Configure (InetSocketAddress (rxIp, fk.port), fk.tcp, payload, subChunks, 65536,
                      &fk.sched, run, flowId, bufBytes);
      fk.srvI = srvIdx[fk.server];
      fk.sender = PeekPointer (app);
      sender->AddApplication (app);
      double start = std::max (0.0, (double) fk.sched.front ().first - 0.001);
      app->SetStartTime (Seconds (start));
    }

  FlowMonitorHelper fmh;
  fmh.SetMonitorAttribute ("DelayBinWidth", DoubleValue (binW));
  fmh.SetMonitorAttribute ("JitterBinWidth", DoubleValue (binW));
  Ptr<FlowMonitor> monitor = fmh.InstallAll ();

  double stopT = (double) lastT + 1.0 + drain;

  // ---------------- epoch-level measurement setup (revision 2)
  {
    EpochState &e = g_ep;
    e.capBps = capMbps > 0 ? capMbps * 1e6 : 0.0;
    e.linkBps = srvBps;
    int64_t tMin = rows.front ().t;
    for (const auto &r : rows)
      tMin = std::min (tMin, r.t);
    e.tFirst = tMin;
    e.tEnd = (int64_t) std::floor (stopT);
    // users per (t, server) and reassignments from the schedule
    std::map<std::pair<int64_t, uint32_t>, std::set<uint32_t>> on;
    std::map<uint32_t, std::map<int64_t, uint32_t>> srvOfUser;   // uid -> t -> server
    for (const auto &r : rows)
      {
        on[{r.t, srvIdx[r.server]}].insert (r.uid);
        srvOfUser[r.uid][r.t] = srvIdx[r.server];
      }
    for (const auto &kv : on)
      e.usersOn[kv.first.first].push_back ({kv.first.second, (uint32_t) kv.second.size ()});
    for (const auto &uk : srvOfUser)
      {
        const auto &seq = uk.second;
        for (auto it = seq.begin (); it != seq.end (); ++it)
          {
            auto nx = std::next (it);
            if (nx == seq.end ())
              break;
            e.nPairs++;
            if (nx->second != it->second)
              e.reassign[nx->first].push_back ({uk.first, it->second});
          }
      }
    e.keyFirst.resize (g_keys.size ());
    e.keyLast.resize (g_keys.size ());
    e.keyRes.assign (g_keys.size (), 0);
    for (uint32_t ki = 0; ki < g_keys.size (); ++ki)
      {
        e.keyFirst[ki] = g_keys[ki].sched.front ().first;
        e.keyLast[ki] = g_keys[ki].sched.back ().first;
        e.keyOrder.push_back (ki);
        e.keysOf[{g_keys[ki].uid, g_keys[ki].srvI}].push_back (ki);
      }
    std::sort (e.keyOrder.begin (), e.keyOrder.end (),
               [&] (uint32_t a, uint32_t b) { return e.keyFirst[a] < e.keyFirst[b]; });
    // capacity checks
    if (!checksFile.empty ())
      {
        std::ifstream cf (checksFile);
        if (!cf)
          {
            std::cerr << "cannot open " << checksFile << "\n";
            return 1;
          }
        std::string cl;
        std::getline (cf, cl); // t,server,load_bps,d_bps,resid_bits,passed
        while (std::getline (cf, cl))
          {
            if (!cl.empty () && cl.back () == '\r')
              cl.pop_back ();
            auto tk = SplitCsv (cl);
            if (tk.size () != 6)
              continue;
            Check c;
            auto sit = srvIdx.find (tk[1]);
            c.srv = (sit == srvIdx.end ()) ? UINT32_MAX : sit->second;
            c.loadBps = std::stod (tk[2]);
            c.dBps = std::stod (tk[3]);
            c.residEstBits = std::stod (tk[4]);
            c.passed = (tk[5] == "1");
            e.checks[std::stoll (tk[0])].push_back (c);
          }
      }
    Simulator::Schedule (Seconds ((double) e.tFirst), &EpochTick, e.tFirst);
  }
  if (verbose)
    {
      std::cout << "users=" << users.GetN () << " servers=" << servers.GetN ()
                << " flowKeys=" << g_keys.size () << " rows=" << rows.size ()
                << " cap=" << (capMbps > 0 ? std::to_string (capMbps) + "Mbps" : std::string ("unlimited"))
                << " run=" << run << " stop=" << stopT << "s" << std::endl;
      g_wall0 = wall0;
      for (int i = 1; i < 10; ++i)
        Simulator::Schedule (Seconds (stopT * i / 10.0), &Progress, i);
    }
  Simulator::Stop (Seconds (stopT));
  Simulator::Run ();

  // ---------------- collect
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> cls = DynamicCast<Ipv4FlowClassifier> (fmh.GetClassifier ());
  std::map<std::pair<std::string, std::string>, Acc> acc;
  for (const auto &kv : monitor->GetFlowStats ())
    {
      Ipv4FlowClassifier::FiveTuple ft = cls->FindFlow (kv.first);
      auto pit = portToKey.find (ft.destinationPort);
      if (pit == portToKey.end ())
        continue; // TCP ACK direction
      const FlowKey &fk = g_keys[pit->second];
      const FlowMonitor::FlowStats &s = kv.second;
      for (const auto &g : GroupsOf (fk.app))
        for (const std::string &d : {std::string ("ALL"), std::string (fk.dl ? "DL" : "UL")})
          {
            Acc &a = acc[{g, d}];
            a.nFlows++;
            a.txPkts += s.txPackets;
            a.rxPkts += s.rxPackets;
            a.txBytes += s.txBytes;
            a.rxBytes += s.rxBytes;
            a.delaySum += s.delaySum.GetSeconds ();
            a.jitterSum += s.jitterSum.GetSeconds ();
            if (s.rxPackets > 1)
              a.jitterN += s.rxPackets - 1;
            double dur = s.timeLastRxPacket.GetSeconds () - s.timeFirstTxPacket.GetSeconds ();
            if (s.rxPackets > 1 && dur > 0)
              {
                a.thrSum += s.rxBytes * 8.0 / dur / 1e6;
                a.thrN++;
              }
            for (uint32_t b = 0; b < s.delayHistogram.GetNBins (); ++b)
              {
                uint32_t c = s.delayHistogram.GetBinCount (b);
                if (c)
                  a.hist[(int64_t) b] += c;
              }
          }
    }
  for (const auto &fk : g_keys)
    for (const auto &g : GroupsOf (fk.app))
      for (const std::string &d : {std::string ("ALL"), std::string (fk.dl ? "DL" : "UL")})
        {
          Acc &a = acc[{g, d}];
          a.offered += fk.offered;
          a.rxApp += fk.rxApp;
        }
  for (auto &kv : acc)
    kv.second.activeUserSec = activeSec[kv.first].size ();

  // stall proxy (video downlink): lag between when a second's bytes were offered and when the
  // cumulative delivered volume of that user reached them
  std::map<uint32_t, std::map<int64_t, uint64_t>> dlRxPerUser;
  for (const auto &fk : g_keys)
    if (fk.dl && kVideo.count (fk.app))
      for (const auto &kv : fk.rxPerSec)
        dlRxPerUser[fk.uid][kv.first] += kv.second;
  for (const auto &uk : dlOfferedPerUser)
    {
      const auto &off = uk.second;
      const auto &rx = dlRxPerUser[uk.first];
      auto rit = rx.begin ();
      double cumO = 0, cumR = 0;
      int64_t rxSec = -1;
      for (const auto &o : off)
        {
          cumO += (double) o.second;
          double target = cumO * 0.999;
          while (cumR < target && rit != rx.end ())
            {
              cumR += (double) rit->second;
              rxSec = rit->first;
              ++rit;
            }
          double lag;
          if (cumR >= target)
            lag = std::max<double> (0.0, (double) (rxSec - o.first));
          else
            lag = INFINITY;
          for (const auto &g : GroupsOf (appOfUser[uk.first]))
            for (const std::string &d : {std::string ("ALL"), std::string ("DL")})
              {
                Acc &a = acc[{g, d}];
                a.stallDen++;
                if (lag > stallThr)
                  a.stallN++;
                a.lagSum += std::isinf (lag) ? stopT - (double) o.first : lag;
              }
        }
    }

  // ---------------- write
  std::ofstream out (outFile);
  out << "run,cap_mbps,group,dir,n_flows,active_user_s,offered_MB,delivered_MB,delivery_ratio,"
         "goodput_mbps,flow_thr_mbps,tx_pkts,rx_pkts,loss_pct,delay_mean_ms,delay_p50_ms,"
         "delay_p95_ms,delay_p99_ms,jitter_mean_ms,share_delay_le_50ms,share_delay_le_100ms,"
         "stall_frac,lag_mean_s\n";
  out << std::setprecision (6);
  for (const auto &kv : acc)
    {
      const Acc &a = kv.second;
      auto nanIf = [] (bool c, double v) { return c ? v : NAN; };
      out << run << "," << capMbps << "," << kv.first.first << "," << kv.first.second << ","
          << a.nFlows << "," << a.activeUserSec << "," << a.offered / 1e6 << "," << a.rxApp / 1e6
          << "," << nanIf (a.offered > 0, (double) a.rxApp / (double) a.offered) << ","
          << nanIf (a.activeUserSec > 0, a.rxApp * 8.0 / 1e6 / (double) a.activeUserSec) << ","
          << nanIf (a.thrN > 0, a.thrSum / a.thrN) << "," << a.txPkts << "," << a.rxPkts << ","
          << nanIf (a.txPkts > 0, 100.0 * (double) (a.txPkts - std::min (a.txPkts, a.rxPkts)) / (double) a.txPkts)
          << "," << nanIf (a.rxPkts > 0, 1000.0 * a.delaySum / (double) a.rxPkts) << ","
          << 1000.0 * HistQuantile (a.hist, 0.50, binW) << ","
          << 1000.0 * HistQuantile (a.hist, 0.95, binW) << ","
          << 1000.0 * HistQuantile (a.hist, 0.99, binW) << ","
          << nanIf (a.jitterN > 0, 1000.0 * a.jitterSum / (double) a.jitterN) << ","
          << HistShareBelow (a.hist, 0.050, binW) << "," << HistShareBelow (a.hist, 0.100, binW)
          << "," << nanIf (a.stallDen > 0, (double) a.stallN / (double) a.stallDen) << ","
          << nanIf (a.stallDen > 0, a.lagSum / (double) a.stallDen) << "\n";
    }
  out.close ();

  // ---------------- write server-level metrics (revision 2)
  {
    EpochState &e = g_ep;
    std::vector<double> perSrv;
    for (const auto &m : g_srv)
      if (m.totBits > 0)
        perSrv.push_back (m.totBits);
    double s1 = 0, s2 = 0, dropB = 0;
    for (double x : perSrv)
      {
        s1 += x;
        s2 += x * x;
      }
    for (const auto &m : g_srv)
      dropB += (double) m.totDropBytes;
    double umean = 0;
    for (double x : e.utils)
      umean += x;
    umean = e.utils.empty () ? NAN : umean / e.utils.size ();
    double rmean = 0, rgt = 0;
    for (double x : e.residAssigned)
      {
        rmean += x;
        rgt += (x > 1.0);
      }
    size_t nr = e.residAssigned.size ();
    auto div = [] (double a, double b) { return b > 0 ? a / b : NAN; };
    std::ofstream so (outSrv);
    so << "run,cap_mbps,n_active_epochs,load_var_util,load_var_util_2plus,util_mean,util_p95,util_max,"
          "jain_server_bytes,n_servers_used,overloaded_server_epoch_frac,overloaded_user_seconds_frac,"
          "dropped_MB,resid_mean_epochs,resid_p95_epochs,resid_gt1_share,n_checks,n_checks_passed,"
          "false_admissions,false_admission_rate,n_checks_failed,false_rejections,resid_est_abs_err_epochs,"
          "n_user_second_pairs,n_reassign,reassign_rate,n_reassign_draining,reassign_draining_share,"
          "drain_MB_mean\n";
    so << std::setprecision (6);
    so << run << "," << capMbps << "," << e.nActiveEpochs << "," << div (e.sumVar, e.nActiveEpochs) << ","
       << div (e.sumVar2, e.nEpochs2plus) << "," << umean << "," << Quantile (e.utils, 0.95) << ","
       << Quantile (e.utils, 1.0) << "," << (perSrv.empty () ? NAN : s1 * s1 / (perSrv.size () * s2)) << ","
       << perSrv.size () << "," << div (e.nOverSrvEpochs, e.nActiveSrvEpochs) << ","
       << div (e.overUserSec, e.userSec) << "," << dropB / 1e6 << "," << (nr ? rmean / nr : NAN) << ","
       << Quantile (e.residAssigned, 0.95) << "," << (nr ? rgt / nr : NAN) << "," << e.nChecks << ","
       << e.nPassed << "," << e.nFalseAdm << "," << div (e.nFalseAdm, e.nPassed) << "," << e.nFailed << ","
       << e.nFalseRej << "," << div (e.sumEstErr, e.nChecks) << "," << e.nPairs << "," << e.nReassign << ","
       << div (e.nReassign, e.nPairs) << "," << e.nReassignDrain << "," << div (e.nReassignDrain, e.nReassign)
       << "," << div (e.drainBytes / 1e6, e.nReassignDrain) << "\n";
    so.close ();
  }

  const Acc &all = acc[{"ALL", "ALL"}];
  double el = std::chrono::duration<double> (std::chrono::steady_clock::now () - wall0).count ();
  std::cout << std::fixed << std::setprecision (3) << "DONE run=" << run
            << " delivery=" << (all.offered ? (double) all.rxApp / all.offered : 0)
            << " delay_mean_ms=" << (all.rxPkts ? 1000.0 * all.delaySum / all.rxPkts : 0)
            << " loss_pct=" << (all.txPkts ? 100.0 * (double) (all.txPkts - std::min (all.txPkts, all.rxPkts)) / all.txPkts : 0)
            << " false_adm=" << g_ep.nFalseAdm << "/" << g_ep.nPassed
            << " wall_s=" << std::setprecision (1) << el << " -> " << outFile << std::endl;

  Simulator::Destroy ();
  return 0;
}
