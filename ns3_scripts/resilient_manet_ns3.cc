// Resilient MANET NS-3 simulation: 100 nodes, 802.11b, RandomWaypoint
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ResilientManet");

int main(int argc, char *argv[]) {
    uint32_t nNodes = 100;
    double duration = 40.0;
    uint32_t seed = 42;

    CommandLine cmd;
    cmd.AddValue("nNodes", "Number of nodes", nNodes);
    cmd.AddValue("duration", "Simulation duration", duration);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);

    NodeContainer nodes;
    nodes.Create(nNodes);

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"));
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=1|Max=10]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
        "PositionAllocator", PointerValue(
            CreateObject<RandomRectanglePositionAllocator>()));
    mobility.Install(nodes);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::AarfWifiManager");

    WifiMacHelper mac;
    Ssid ssid = Ssid("resilient-manet");
    mac.SetType("ns3::AdhocWifiMac", "Ssid", SsidValue(ssid));

    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    uint16_t port = 9;
    OnOffHelper onoff("ns3::UdpSocketFactory",
        Address(InetSocketAddress(interfaces.GetAddress(nNodes-1), port)));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    onoff.SetAttribute("DataRate", StringValue("100kbps"));
    onoff.SetAttribute("PacketSize", UintegerValue(512));

    ApplicationContainer apps = onoff.Install(nodes.Get(0));
    apps.Start(Seconds(1.0));
    apps.Stop(Seconds(duration));

    PacketSinkHelper sink("ns3::UdpSocketFactory",
        Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    apps.Add(sink.Install(nodes.Get(nNodes-1)));
    apps.Start(Seconds(1.0));
    apps.Stop(Seconds(duration));

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(duration + 1.0));
    Simulator::Run();

    monitor->SerializeToXmlFile("resilient_manet_flowmon.xml", true, true);
    Simulator::Destroy();
    return 0;
}
