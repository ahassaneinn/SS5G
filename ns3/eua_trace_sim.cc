/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * eua_trace_sim.cc  --  edge-server-level, trace-driven ns-3 evaluation of edge user allocation
 *
 * The allocation (which server serves which user in every second, incl. load balancing)
 * is computed beforehand by allocate.py; this program replays that schedule packet by packet.
 *
 * Topology (star, core/edge level -- NOT a 5G radio model; see README "Scope"):
 *
 *     user_i --[access link: accessRate, one-way delay ~U(accessMinMs, accessMaxMs) per user]--+
 *                                                                                            GW
 *     server_j --[server link: capacity = --cap Mbps, delay = great-circle(GW,server)*stretch/c_fiber]--+
 *
 *   * The per-server capacity is a real bottleneck (link rate + finite queue).
 *   * Propagation delay comes from the geographic distance in km.
 *   * Traffic = the dominant-server traffic of each TEST-period user-second of the trace
 *     (identical bytes for every allocator), sent in the second it was observed.
 *     TCP: the second's bytes are pushed in --subChunks pieces at random offsets inside the second
 *     (application-limited TCP; backlog carries over).  UDP: paced packets with random gaps.
 *   * Randomness (--run): per-user access delay, intra-second timing. Use several --run values
 *     per configuration.
 *
 * Metrics (per group = ALL / VIDEO / GAMING / each app, per direction = ALL / DL / UL):
 *   offered / delivered bytes, delivery ratio, goodput per active user-second, per-flow
 *   throughput, packet loss, one-way delay mean / p50 / p95 / p99 (FlowMonitor histograms),
 *   jitter, share of packets under 50 / 100 ms, and for video downlink a stall proxy:
 *   share of offered seconds whose bytes are not fully delivered within --stallThr seconds.
 *
 * Build: copy to <ns-3>/scratch/ and run "./ns3 build scratch/eua_trace_sim"  (ns-3.36 or newer)
 * Run  : ./ns3 run "eua_trace_sim --flows=flows.csv --cap=10 --run=1 --out=res.csv"
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
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("EuaTraceSim");

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
             const std::vector<std::pair<int64_t, uint64_t>> *sched, Ptr<UniformRandomVariable> rng)
  {
    m_peer = peer;
    m_tcp = tcp;
    m_udpPayload = udpPayload;
    m_subChunks = std::max<uint32_t> (1, subChunks);
    m_writeSize = writeSize;
    m_sched = sched;
    m_rng = rng;
  }

  uint64_t Pending () const { return m_pending; }

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
                double off = (j + m_rng->GetValue (0.0, 1.0)) / m_subChunks;
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
            double off = m_rng->GetValue (0.0, gap);
            m_events.push_back (Simulator::Schedule (Seconds (t0 + off), &TraceSender::UdpBurst,
                                                     this, n, last, gap));
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
  UdpBurst (uint32_t left, uint32_t lastSize, double gap)
  {
    uint32_t sz = (left == 1) ? lastSize : m_udpPayload;
    m_socket->Send (Create<Packet> (sz));
    if (left > 1)
      {
        double g = gap * (0.5 + m_rng->GetValue (0.0, 1.0)); // mean gap preserved
        Simulator::Schedule (Seconds (g), &TraceSender::UdpBurst, this, left - 1, lastSize, gap);
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
  Ptr<UniformRandomVariable> m_rng;
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
          uint32_t qPkts, Ipv4AddressHelper &addr)
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
  tch.Install (d);

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
int
main (int argc, char *argv[])
{
  auto wall0 = std::chrono::steady_clock::now ();

  std::string flowsFile = "flows.csv";
  std::string outFile = "eua_result.csv";
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
  cmd.Parse (argc, argv);

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

  Ptr<UniformRandomVariable> accRng = CreateObject<UniformRandomVariable> ();
  std::vector<Ipv4Address> userIp (users.GetN ()), srvIp (servers.GetN ());
  double accessBps = accessMbps * 1e6;
  for (const auto &kv : userIdx) // ascending uid -> same delays for every allocator (same run)
    {
      double dMs = accRng->GetValue (accessMinMs, accessMaxMs);
      auto ifc = MakeLink (users.Get (kv.second), gw.Get (0), accessBps, dMs, qdisc,
                           QueuePackets (accessBps, bufferMs, minQ, maxQ), addr);
      userIp[kv.second] = ifc.GetAddress (0);
    }
  double srvBps = (capMbps > 0 ? capMbps : unlimitedMbps) * 1e6;
  for (const auto &kv : srvIdx)
    {
      auto ifc = MakeLink (servers.Get (kv.second), gw.Get (0), srvBps, srvDelay[kv.first], qdisc,
                           QueuePackets (srvBps, bufferMs, minQ, maxQ), addr);
      srvIp[kv.second] = ifc.GetAddress (0);
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
  Ptr<UniformRandomVariable> appRng = CreateObject<UniformRandomVariable> ();
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
      app->Configure (InetSocketAddress (rxIp, fk.port), fk.tcp, payload, subChunks, 65536,
                      &fk.sched, appRng);
      sender->AddApplication (app);
      double start = std::max (0.0, (double) fk.sched.front ().first - 0.001);
      app->SetStartTime (Seconds (start));
    }

  FlowMonitorHelper fmh;
  fmh.SetMonitorAttribute ("DelayBinWidth", DoubleValue (binW));
  fmh.SetMonitorAttribute ("JitterBinWidth", DoubleValue (binW));
  Ptr<FlowMonitor> monitor = fmh.InstallAll ();

  double stopT = (double) lastT + 1.0 + drain;
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

  const Acc &all = acc[{"ALL", "ALL"}];
  double el = std::chrono::duration<double> (std::chrono::steady_clock::now () - wall0).count ();
  std::cout << std::fixed << std::setprecision (3) << "DONE run=" << run
            << " delivery=" << (all.offered ? (double) all.rxApp / all.offered : 0)
            << " delay_mean_ms=" << (all.rxPkts ? 1000.0 * all.delaySum / all.rxPkts : 0)
            << " loss_pct=" << (all.txPkts ? 100.0 * (double) (all.txPkts - std::min (all.txPkts, all.rxPkts)) / all.txPkts : 0)
            << " wall_s=" << std::setprecision (1) << el << " -> " << outFile << std::endl;

  Simulator::Destroy ();
  return 0;
}
