/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2017 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License version 2 as
 *   published by the Free Software Foundation;
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
*/

#include "ns3/core-module.h"
#include "ns3/nr-helper.h"
#include "ns3/nr-point-to-point-epc-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/eps-bearer-tag.h"
#include "ns3/grid-scenario-helper.h"
#include "ns3/log.h"
#include "ns3/antenna-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/point-to-point-helper.h"
#include <cmath>

// ---------------------------

#include "ns3/core-module.h"
#include "ns3/config-store.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/nr-module.h"
#include "ns3/config-store-module.h"
#include "ns3/antenna-module.h"
#include "ns3/netanim-module.h"
#include "ns3/buildings-module.h"



/**
 * \file cttc-3gpp-channel-nums.cc
 * \ingroup examples
 * \brief Simple topology numerologies example.
 *
 * This example allows users to configure the numerology and test the end-to-end
 * performance for different numerologies. In the following figure we illustrate
 * the simulation setup.
 *
 * For example, UDP packet generation rate can be configured by setting
 * "--lambda=1000". The numerology can be toggled by the argument,
 * e.g. "--numerology=1". Additionally, in this example two arguments
 * are added "bandwidth" and "frequency", both in Hz units. The modulation
 * scheme of this example is in test mode, and it is fixed to 28.
 *
 * By default, the program uses the 3GPP channel model, without shadowing and with
 * line of sight ('l') option. The program runs for 0.4 seconds and one single
 * packet is to be transmitted. The packet size can be configured by using the
 * following parameter: "--packetSize=1000".
 *
 * This simulation prints the output to the terminal and also to the file which
 * is named by default "cttc-3gpp-channel-nums-fdm-output" and which is by
 * default placed in the root directory of the project.
 *
 * To run the simulation with the default configuration one shall run the
 * following in the command line:
 *
 * ./waf --run cttc-3gpp-channel-nums
 *
 */



using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LenaSimpleEpc");






// // ---------------------------------------------------------------------------

int main (int argc, char *argv[]){

    // normal logging code
    bool logging=true;

    // set simulation time and mobility
    double simTime=5;//seconds
    double udpAppStartTime = 0.4; //seconds

    //other simulation parameters default values
    uint16_t numerology = 1;
    // uint16_t numerology = 2;


    //uint32_t PacketSize = 1000;

    uint16_t gNbNum = 1; //according to given topology configurations
    uint16_t ueNumPergNb = 11; //according to given topology configurations
    uint16_t numBearersPerUe=1;
    double centralFrequency = 6e9; //according to given topology configurations
    double bandwidth = 10e6; //according to given topology configurations
    double txPower = 23; //according to given topology configurations
    double lambda = 1000;//by default set to non-full buffer case

    // double lambdaUll=1000;

    uint32_t udpPacketSize = 1400; //according to given topology configurations
    bool udpFullBuffer = true;
    uint8_t fixedMcs = 28;
    bool useFixedMcs = false;
    bool singleUeTopology = false;
    uint16_t RngRun = 30;

    // double speed=10;
    bool mobile=true;
    
    std::string speed = "ns3::ConstantRandomVariable[Constant=10.0]";



    // Where we will store the output files.
    std::string simTag = "outputFile";
    std::string outputDir = "../../../scratch/nr_bsr_allUE/flow/";
    std::string pattern = "F|F|F|F|F|F|F|F|F|F|";


    CommandLine cmd;
    cmd.AddValue ("gNbNum",
                "The number of gNbs in multiple-ue topology",
                gNbNum);
    cmd.AddValue ("ueNumPergNb",
                "The number of UE per gNb in multiple-ue topology",
                ueNumPergNb);
    cmd.AddValue ("logging",
                "Enable logging",
                logging);
    cmd.AddValue ("numerology",
                "The numerology to be used.",
                numerology);
    cmd.AddValue ("txPower",
                "Tx power to be configured to gNB",
                txPower);
    cmd.AddValue ("simTag",
                "tag to be appended to output filenames to distinguish simulation campaigns",
                simTag);
    cmd.AddValue ("outputDir",
                "directory where to store simulation results",
                outputDir);
    cmd.AddValue ("frequency",
                "The system frequency",
                centralFrequency);
    cmd.AddValue ("bandwidth",
                "The system bandwidth",
                bandwidth);
    cmd.AddValue ("udpPacketSize",
                "UDP packet size in bytes",
                udpPacketSize);
    cmd.AddValue ("lambda",
                "Number of UDP packets per second",
                lambda);
    cmd.AddValue ("udpFullBuffer",
                "Whether to set the full buffer traffic; if this parameter is set then the udpInterval parameter"
                "will be neglected",
                udpFullBuffer);
    cmd.AddValue ("fixedMcs",
                "The fixed MCS that will be used in this example if useFixedMcs is configured to true (1).",
                fixedMcs);
    cmd.AddValue ("useFixedMcs",
                "Whether to use fixed mcs, normally used for testing purposes",
                useFixedMcs);
    cmd.AddValue ("singleUeTopology",
                "If true, the example uses a predefined topology with one UE and one gNB; "
                "if false, the example creates a grid of gNBs with a number of UEs attached",
                singleUeTopology);
    cmd.AddValue ("mobile","If true , then mobile case of UE,if false = then static case",mobile);
    // cmd.AddValue ("scheduler",
    //               "Scheduler to be used",
    //               scheduler);
    cmd.AddValue ("speed","Set speed for mobility case",speed);
    cmd.Parse (argc, argv);


    LogLevel logLevel1 = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_PREFIX_NODE | LOG_LEVEL_ALL);
    if (logging)
    {
        // LogComponentEnable ("ApplicationContainer",logLevel1);
        // LogComponentEnable("OnOffApplication",logLevel1);
        // LogComponentEnable ("UdpClient", LOG_LEVEL_INFO);
        // LogComponentEnable ("UdpServer", LOG_LEVEL_INFO);
        LogComponentEnable ("NrUeMac", logLevel1);  //keep


        // LogComponentEnable ("MyTdmaPF", logLevel1);
        // LogComponentEnable ("MyTdmaRR", logLevel1);
        // LogComponentEnable ("MyTdma", logLevel1);
        // LogComponentEnable ("NrMacSchedulerNs3", logLevel1);

    }

    RngSeedManager::SetSeed(5);
    RngSeedManager::SetRun(RngRun);

    std::string exampleName = simTag + "-" + "nr-v2x-simple-demo";

    if(numerology==3)
    centralFrequency=28e9;

    if(udpFullBuffer)
    lambda=2500;

    // if(mobile)
    // simTime = 20;//simTime for Mobility case (seconds)

    NS_ASSERT (ueNumPergNb > 0);

    // setup the nr simulation
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper> ();

    /*
    * Spectrum division. We create one operation band with one component carrier
    * (CC) which occupies the whole operation band bandwidth. The CC contains a
    * single Bandwidth Part (BWP). This BWP occupies the whole CC band.
    * Both operational bands will use the StreetCanyon channel modeling.
    */
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;  // in this example, both bands have a single CC
    BandwidthPartInfo::Scenario scenario = BandwidthPartInfo::UMa_LoS;



    // Create the configuration for the CcBwpHelper. SimpleOperationBandConf creates
    // a single BWP per CC
    CcBwpCreator::SimpleOperationBandConf bandConf (centralFrequency,
                                                    bandwidth,
                                                    numCcPerBand,
                                                    scenario);

    // By using the configuration created, it is time to make the operation bands
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc (bandConf);

    /*
    * Initialize channel and pathloss, plus other things inside band1. If needed,
    * the band configuration can be done manually, but we leave it for more
    * sophisticated examples. For the moment, this method will take care
    * of all the spectrum initialization needs.
    */
    nrHelper->InitializeOperationBand (&band);

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps ({band});

    /*
    * Continue setting the parameters which are common to all the nodes, like the
    * gNB transmit power or numerology.
    */
    nrHelper->SetGnbPhyAttribute ("TxPower", DoubleValue (txPower));
    nrHelper->SetGnbPhyAttribute ("Numerology", UintegerValue (numerology));

    // Scheduler

    Config::SetDefault ("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue (999999999));

    nrHelper->SetSchedulerTypeId (TypeId::LookupByName ("ns3::MyTdmaPF"));

    // Antennas for all the UEs
    nrHelper->SetUeAntennaAttribute ("NumRows", UintegerValue (2));
    nrHelper->SetUeAntennaAttribute ("NumColumns", UintegerValue (4));
    nrHelper->SetUeAntennaAttribute ("AntennaElement", PointerValue (CreateObject<IsotropicAntennaModel> ()));

    // Antennas for all the gNbs
    nrHelper->SetGnbAntennaAttribute ("NumRows", UintegerValue (4));
    nrHelper->SetGnbAntennaAttribute ("NumColumns", UintegerValue (8));
    nrHelper->SetGnbAntennaAttribute ("AntennaElement", PointerValue (CreateObject<ThreeGppAntennaModel> ()));

    // Beamforming method
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    idealBeamformingHelper->SetAttribute ("BeamformingMethod", TypeIdValue (DirectPathBeamforming::GetTypeId ()));
    nrHelper->SetBeamformingHelper (idealBeamformingHelper);

    Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod",TimeValue (MilliSeconds (0)));
    //  nrHelper->SetChannelConditionModelAttribute ("UpdatePeriod", TimeValue (MilliSeconds (0)));
    nrHelper->SetPathlossAttribute ("ShadowingEnabled", BooleanValue (false));

    // Error Model: UE and GNB with same spectrum error model.
    nrHelper->SetUlErrorModel ("ns3::NrEesmIrT1");
    nrHelper->SetDlErrorModel ("ns3::NrEesmIrT1");

    // Both DL and UL AMC will have the same model behind.

    nrHelper->SetGnbDlAmcAttribute ("AmcModel", EnumValue (NrAmc::ShannonModel)); // NrAmc::ShannonModel or NrAmc::ErrorModel
    nrHelper->SetGnbUlAmcAttribute ("AmcModel", EnumValue (NrAmc::ShannonModel)); // NrAmc::ShannonModel or NrAmc::ErrorModel


    // Create EPC helper
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper> ();
    nrHelper->SetEpcHelper (epcHelper);
    // Core latency
    epcHelper->SetAttribute ("S1uLinkDelay", TimeValue (MilliSeconds (2)));

    // gNb routing between Bearer and bandwidh part
    uint32_t bwpIdForBearer = 0;
    nrHelper->SetGnbBwpManagerAlgorithmAttribute ("GBR_CONV_VOICE", UintegerValue (bwpIdForBearer));

    // Initialize nrHelper
    nrHelper->Initialize ();


    /*
    *  Create the gNB and UE nodes according to the network topology
    */
    NodeContainer gNbNodes;
    NodeContainer ueNodes;
    MobilityHelper mobility;

    // if(mobile)
    //   mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel","Speed", StringValue (speed),
    //                            "Bounds", StringValue ("-3000|3000|-3000|3000"));
    // else
    //     mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

    Ptr<ListPositionAllocator> bsPositionAlloc = CreateObject<ListPositionAllocator> ();
    Ptr<ListPositionAllocator> utPositionAlloc = CreateObject<ListPositionAllocator> ();

    const double gNbHeight = 10;
    const double ueHeight = 1.5;

    if (singleUeTopology)
    {
        gNbNodes.Create (1);
        ueNodes.Create (1);
        gNbNum = 1;
        ueNumPergNb = 1;

        mobility.Install (gNbNodes);
        mobility.Install (ueNodes);
        bsPositionAlloc->Add (Vector (0.0, 0.0, gNbHeight));
        utPositionAlloc->Add (Vector (0.0, 30.0, ueHeight));
    }
    else
    {
        gNbNodes.Create (gNbNum);
        ueNodes.Create (ueNumPergNb * gNbNum);

        int32_t yValue = 0.0;
        for (uint32_t i = 1; i <= gNbNodes.GetN (); ++i)
        {
            // 2.0, -2.0, 6.0, -6.0, 10.0, -10.0, ....
            if (i % 2 != 0)
            {
                yValue = static_cast<int> (i) * 30;
            }
            else
            {
                yValue = -yValue;
            }

            bsPositionAlloc->Add (Vector (0.0, yValue, gNbHeight));

            // 1.0, -1.0, 3.0, -3.0, 5.0, -5.0, ...
            double xValue = 0.0;
            for (uint16_t j = 1; j <= ueNumPergNb; ++j)
            {
                if (j % 2 != 0)
                {
                    xValue = j;
                }
                else
                {
                    xValue = -xValue;
                }

                if (yValue > 0)
                {
                    utPositionAlloc->Add (Vector (xValue, 1, ueHeight));
                }
                else
                {
                    utPositionAlloc->Add (Vector (xValue, -1, ueHeight));
                }
            }
        }
    }


    mobility.SetPositionAllocator (bsPositionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (gNbNodes);
    
    mobility.SetPositionAllocator (utPositionAlloc);
    // mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel","Speed", StringValue (speed), "Bounds", StringValue ("-3000|3000|-3000|3000"));
    mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel","Speed", StringValue ("ns3::UniformRandomVariable[Min=5.0|Max=15.0]"), "Bounds", StringValue ("-500|500|-500|500"));
    mobility.Install (ueNodes);




    // Install nr net devices
    NetDeviceContainer gNbNetDev = nrHelper->InstallGnbDevice (gNbNodes,
                                                                allBwps);

    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice (ueNodes,
                                                            allBwps);
    // Band40: CC0 - BWP0 & Band38: CC1 - BWP1
    nrHelper->GetGnbPhy (gNbNetDev.Get (0), 0)->SetAttribute ("Numerology", UintegerValue (numerology));
    nrHelper->GetGnbPhy (gNbNetDev.Get (0), 0)->SetAttribute ("TxPower",DoubleValue (txPower));
    nrHelper->GetGnbPhy (gNbNetDev.Get (0), 0)->SetAttribute ("Pattern", StringValue (pattern));
    nrHelper->GetGnbPhy (gNbNetDev.Get (0), 0)->SetAttribute ("RbOverhead", DoubleValue (0.1));

    int64_t randomStream = 1;
    randomStream += nrHelper->AssignStreams (gNbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams (ueNetDev, randomStream);


    // When all the configuration is done, explicitly call UpdateConfig ()

    for (auto it = gNbNetDev.Begin (); it != gNbNetDev.End (); ++it)
    {
        DynamicCast<NrGnbNetDevice> (*it)->UpdateConfig ();
    }

    for (auto it = ueNetDev.Begin (); it != ueNetDev.End (); ++it)
    {
        DynamicCast<NrUeNetDevice> (*it)->UpdateConfig ();
    }

    // create the internet and install the IP stack on the UEs
    // get SGW/PGW and create a single RemoteHost
    Ptr<Node> pgw = epcHelper->GetPgwNode ();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);

    // connect a remoteHost to pgw. Setup routing too
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("10Gb/s")));
    p2ph.SetDeviceAttribute ("Mtu", UintegerValue (2500));
    // p2ph.SetChannelAttribute ("Delay", StringValue (bottleneck_delay));
    p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.0001)));

    NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
    remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);
    internet.Install (ueNodes);


    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueNetDev));

    // Set the default gateway for the UEs
    for (uint32_t j = 0; j < ueNodes.GetN (); ++j)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNodes.Get (j)->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    // attach UEs to the closest eNB
    nrHelper->AttachToClosestEnb (ueNetDev, gNbNetDev);

    //--------------------------Install and start applications on UEs and remote host-------------------Starts Here----------------------//
    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;

    // randomize a bit start times to avoid simulation artifacts
    // (e.g., buffer overflows due to packet transmissions happening exactly at the same time)

    Ptr<UniformRandomVariable> startTimeSeconds = CreateObject<UniformRandomVariable> ();
    startTimeSeconds->SetAttribute ("Min", DoubleValue (0));
    startTimeSeconds->SetAttribute ("Max", DoubleValue (0.010));

    for (int u = 0; u < ueNumPergNb; ++u)
    {
        Ptr<Node> ue = ueNodes.Get (u);
        // Set the default gateway for the UE


        for (uint32_t b = 0; b < numBearersPerUe; ++b)
        {
            ++dlPort;
            ++ulPort;

            UdpClientHelper ulClient (remoteHostAddr, ulPort);
            ulClient.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
            // ulClient.SetAttribute ("Interval", TimeValue (Seconds(1.0/lambdaUll)));
            ulClient.SetAttribute ("Interval", TimeValue (Seconds(0.002)));
            ulClient.SetAttribute ("MaxPackets", UintegerValue(0xFFFFFFFF));
            
            
            ApplicationContainer appsul = ulClient.Install(ue); //install in n0
            PacketSinkHelper sink2 ("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), ulPort)));
            appsul = sink2.Install(remoteHost);
            
            //clientApps.Add (ulClient.Install (gridScenario.GetUserTerminals().Get(u)));

            Ptr<EpcTft> tft = Create<EpcTft> ();
            EpcTft::PacketFilter ulpf;
            ulpf.remotePortStart = ulPort;
            ulpf.remotePortEnd = ulPort;
            tft->Add (ulpf);
            EpsBearer bearer (EpsBearer::GBR_CONV_VOICE); 
            nrHelper->ActivateDedicatedEpsBearer (ueNetDev.Get (u), bearer, tft); //CHANGE
            Time startTime = Seconds (startTimeSeconds->GetValue ());
            //Time startTime = Seconds (0.5);
            appsul.Start (startTime);
        
        
        
        
        
    /*   
        // OnOffHelper onoff ("ns3::UdpSocketFactory", Address (InetSocketAddress (ueIpIface.GetAddress (u), dlPort))); // (the address of the remote node to send traffic to. in this case n3)
            // onoff.SetAttribute("OnTime",StringValue("ns3::ConstantRandomVariable[Constant=30]"));
            // onoff.SetAttribute("OffTime",StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            // ApplicationContainer appsdl = onoff.Install(remoteHost); //install in n0
            // PacketSinkHelper sink1 ("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), dlPort)));
            // appsdl = sink1.Install(ue);

            
            OnOffHelper onoff2 ("ns3::UdpSocketFactory", Address (InetSocketAddress (remoteHostAddr, ulPort))); // (the address of the remote node to send traffic to. in this case n3)
            onoff2.SetAttribute("OnTime",  StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onoff2.SetAttribute("OffTime",StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff2.SetAttribute("PacketSize", UintegerValue(1000));
        onoff2.SetAttribute ("DataRate", DataRateValue (DataRate ("3Mbps")));
            ApplicationContainer appsul = onoff2.Install(ue); //install in n0
            PacketSinkHelper sink2 ("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), ulPort)));
            appsul = sink2.Install(remoteHost);

                Ptr<EpcTft> tft = Create<EpcTft> (); //packet filter setup - by default any traffic is allowed
            // EpcTft::PacketFilter dlpf;
            // dlpf.localPortStart = dlPort;
            // dlpf.localPortEnd = dlPort;
            // tft->Add (dlpf);
        

        EpcTft::PacketFilter ulpf;
            ulpf.remotePortStart = ulPort;
            ulpf.remotePortEnd = ulPort;
            tft->Add (ulpf);
            EpsBearer bearer (EpsBearer::GBR_CONV_VOICE); 
            nrHelper->ActivateDedicatedEpsBearer (ueNetDev.Get (u), bearer, tft); //CHANGE

            Time startTime = Seconds (startTimeSeconds->GetValue ());
            appsul.Start (startTime);
            //appsdl.Start (startTime);

    */   


        }
    }


    //enable the traces provided by the nr module
    nrHelper->EnableTraces();


    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add (remoteHost);
    endpointNodes.Add (ueNodes);


    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install (endpointNodes);
    monitor->SetAttribute ("DelayBinWidth", DoubleValue (0.001));
    monitor->SetAttribute ("JitterBinWidth", DoubleValue (0.001));
    monitor->SetAttribute ("PacketSizeBinWidth", DoubleValue (20));





    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();


    // Print per-flow statistics
    monitor->CheckForLostPackets ();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier ());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();

    double averageFlowThroughput = 0.0;
    double averageFlowDelay = 0.0;
    double averagePacketLoss=0.0;

    // AnimationInterface anim("hand.xml");

    std::ofstream outFile;
    std::string filename = outputDir + "/" + simTag;
    outFile.open (filename.c_str (), std::ofstream::out | std::ofstream::app);
    if (!outFile.is_open ())
    {
        NS_LOG_ERROR ("Can't open file " << filename);
        return 1;
    }
    outFile.setf (std::ios_base::fixed);

    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
        std::stringstream protoStream;
        protoStream << (uint16_t) t.protocol;
        if (t.protocol == 6)
        {
            protoStream.str ("TCP");
        }
        if (t.protocol == 17)
        {
            protoStream.str ("UDP");
        }
        outFile << "\nFlow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> " << t.destinationAddress << ":" << t.destinationPort << ") proto " << protoStream.str () << "\n";
        outFile << "  Tx Packets: " << i->second.txPackets << "\n";
        outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
        outFile << "  TxOffered:  " << i->second.txBytes * 8.0 / (simTime - udpAppStartTime) / 1000 / 1000  << " Mbps\n";
        outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        if (i->second.rxPackets > 0)
        {
            // Measure the duration of the flow from receiver's perspective
            double rxDuration = i->second.timeLastRxPacket.GetSeconds () - i->second.timeFirstTxPacket.GetSeconds ();

            averageFlowThroughput += i->second.rxBytes * 8.0 / rxDuration / 1000 / 1000;
            averageFlowDelay += 1000 * i->second.delaySum.GetSeconds () / i->second.rxPackets;
            
            outFile << "  Throughput: " << i->second.rxBytes * 8.0 / rxDuration / 1000 / 1000  << " Mbps\n";
            outFile << "  Mean delay:  " << 1000 * i->second.delaySum.GetSeconds () / i->second.rxPackets << " ms\n";
            //outFile << "  Mean upt:  " << i->second.uptSum / i->second.rxPackets / 1000/1000 << " Mbps \n";
            outFile << "  Mean jitter:  " << 1000 * i->second.jitterSum.GetSeconds () / i->second.rxPackets  << " ms\n";
        }
        else
        {
            outFile << "  Throughput:  0 Mbps\n";
            outFile << "  Mean delay:  0 ms\n";
            outFile << "  Mean upt:  0  Mbps \n";
            outFile << "  Mean jitter: 0 ms\n";
        }
        outFile << "  Rx Packets: " << i->second.rxPackets << "\n";
        double packetLoss=i->second.txPackets-i->second.rxPackets ;
        double lossRate=(packetLoss/i->second.txPackets)*100;
        outFile << " Loss Rate: " <<lossRate<<"\n";
        averagePacketLoss+= packetLoss/ i->second.txPackets;

    }




    outFile << "\n\n  Mean flow throughput: " << averageFlowThroughput / stats.size () << "\n";
    outFile << "\n\n  Aggregrate throughput: " << averageFlowThroughput<< "\n";
    outFile << "\n\n  Mean flow delay: " << averageFlowDelay / stats.size () << "\n";
    outFile << "\n\n  Mean packet loss rate(%): " << (averagePacketLoss / stats.size ())*100 << "\n";
    outFile << "\n========================================================================\n";
    outFile.close ();

    /*
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {

    Ptr<UdpClient> clientApp = appsul.Get (i)->GetObject<UdpClient> ();
    Ptr<UdpServer> serverApp = appsdl.Get (0)->GetObject<UdpServer> ();
    std::cout << "\n Total UDP throughput (bps):" <<
    (serverApp->GetReceived () * udpPacketSize * 8) / (simTime - udpAppStartTime) << std::endl;
    }
    */ 


    Simulator::Destroy ();
    return 0;
}

// // ------------------------------------------------------------------------
// // RLC-RLC delay setup

// std::vector<std::pair <uint32_t,float>> numberOfPacketsDelayFromVehiclesAtRLC;
// std::vector<uint32_t> numberOfPacketsSentFromVehicle;
// static bool g_rxRxRlcPDUCallbackCalled = false;
// static bool g_rxPdcpCallbackCalled = false;
// Time delay;


// /*
//  * TraceSink, RxRlcPDU connects the trace sink with the trace source (RxPDU). It connects the UE with gNB and vice versa.
//  */
// void RxRlcPDU (std::string path, uint16_t rnti, uint8_t lcid, uint32_t bytes, uint64_t rlcDelay)
// {
//   g_rxRxRlcPDUCallbackCalled = true;
//   delay = Time::FromInteger(rlcDelay,Time::NS);
//   std::cout<<"\n rlcDelay in NS (Time):"<< delay<<std::endl;

//   std::cout<<"\n\n Data received at RLC layer at: "<<" RNTI: "<<rnti<<" "<<Simulator::Now()<<" Number of Bytes: "<<bytes<<std::endl;

// //   m_ScenarioFile << "\n\n Data received at RLC layer at:" << Simulator::Now () << std::endl;
// //   m_ScenarioFile << "\n rnti:" << rnti << std::endl;
// //   m_ScenarioFile << "\n delay :" << rlcDelay << std::endl;
  
//   numberOfPacketsDelayFromVehiclesAtRLC[rnti].first=numberOfPacketsDelayFromVehiclesAtRLC[rnti].first+1;
//   numberOfPacketsDelayFromVehiclesAtRLC[rnti].second=numberOfPacketsDelayFromVehiclesAtRLC[rnti].second+delay.GetSeconds();
  
// }

// void
// RxPdcpPDU (std::string path, uint16_t rnti, uint8_t lcid, uint32_t bytes, uint64_t pdcpDelay)
// {
//   std::cout << "\n Packet PDCP delay:" << pdcpDelay << "\n";
//   g_rxPdcpCallbackCalled = true;
// }

// void
// ConnectUlPdcpRlcTraces ()
// {
//   Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/UeMap/*/DataRadioBearerMap/*/LteRlc/RxPDU", MakeCallback (&RxRlcPDU));
//   Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/LteEnbRrc/UeMap/*/DataRadioBearerMap/*/LtePdcp/RxPDU", MakeCallback (&RxPdcpPDU));
//   NS_LOG_INFO ("Received PDCP RLC UL");
// }

// // -------------------------------------------------------------------------


    // numberOfPacketsDelayFromVehiclesAtRLC = std::vector<std::pair<uint32_t,float>> (ueNumPergNb+1, std::make_pair(0, 0));
    // numberOfPacketsSentFromVehicle = std::vector<uint32_t> (ueNumPergNb,{0});

//     // Scheduling To Log RLC-RLC Delay
//     Simulator::Schedule (Seconds (.1), &ConnectUlPdcpRlcTraces);

//     // ------------------------------------------------------
//     for(long unsigned int i=1; i<numberOfPacketsDelayFromVehiclesAtRLC.size(); i++)
//     {
// 	    double x = numberOfPacketsDelayFromVehiclesAtRLC[i].first;
//         // double y = numberOfPacketsSentFromVehicle[i-1];
//         double z = numberOfPacketsDelayFromVehiclesAtRLC[i].second;
//         // outFile <<"RNTI:\t"<<i<<"\tRLC Average Delay in MS:\t" <<double(z/x)*1000<<"\n";
//         outFile <<"RNTI:\t"<<i<<"\tNumber of packet: \t"<< x << " Second\t"<< z <<"\tRLC Average Delay in MS:\t" <<double(z/x)*1000<<"\n";
//     }
//     // ---------------------------------------------------------