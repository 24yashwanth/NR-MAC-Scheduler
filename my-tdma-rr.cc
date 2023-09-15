/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
#include "my-tdma-rr.h"
#include "ns3/nr-mac-scheduler-ue-info-rr.h"
#include <ns3/log.h>
#include <algorithm>
#include <functional>

namespace ns3  {

NS_LOG_COMPONENT_DEFINE ("MyTdmaRR");
NS_OBJECT_ENSURE_REGISTERED (MyTdmaRR);

TypeId
MyTdmaRR::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MyTdmaRR")
    .SetParent<MyTdma> ()
    .AddConstructor<MyTdmaRR> ()
  ;
  return tid;
}

MyTdmaRR::MyTdmaRR ()
  : MyTdma ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG("MY LINE");
  m_bsrDl = Create<BSRDL> (1357);
}

std::shared_ptr<NrMacSchedulerUeInfo>
MyTdmaRR::CreateUeRepresentation (const NrMacCschedSapProvider::CschedUeConfigReqParameters &params) const
{
  NS_LOG_FUNCTION (this);
  return std::make_shared <NrMacSchedulerUeInfoRR> (params.m_rnti, params.m_beamId,
                                                        std::bind (&MyTdmaRR::GetNumRbPerRbg, this));
}

void
MyTdmaRR::AssignedDlResources (const UePtrAndBufferReq &ue,
                                               const FTResources &assigned,
                                               const FTResources &totAssigned) const
{
  NS_LOG_FUNCTION (this);
  NS_UNUSED (assigned);
  NS_UNUSED (totAssigned);
  GetFirst GetUe;
  GetUe (ue)->UpdateDlMetric (m_dlAmc);
}

void
MyTdmaRR::AssignedUlResources (const UePtrAndBufferReq &ue,
                                               const FTResources &assigned,
                                               const FTResources &totAssigned) const
{
  NS_LOG_FUNCTION (this);
  NS_UNUSED (assigned);
  NS_UNUSED (totAssigned);
  GetFirst GetUe;
  GetUe (ue)->UpdateUlMetric (m_ulAmc);
}

std::function<bool(const NrMacSchedulerNs3::UePtrAndBufferReq &lhs,
                   const NrMacSchedulerNs3::UePtrAndBufferReq &rhs )>
MyTdmaRR::GetUeCompareDlFn () const
{
  return NrMacSchedulerUeInfoRR::CompareUeWeightsDl;
}

std::function<bool(const NrMacSchedulerNs3::UePtrAndBufferReq &lhs,
                   const NrMacSchedulerNs3::UePtrAndBufferReq &rhs )>
MyTdmaRR::GetUeCompareUlFn () const
{
  return NrMacSchedulerUeInfoRR::CompareUeWeightsUl;
}

} //namespace ns3
