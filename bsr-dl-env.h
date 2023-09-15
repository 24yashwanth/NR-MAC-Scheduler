/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Huazhong University of Science and Technology, Dian Group
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Pengyu Liu <eic_lpy@hust.edu.cn>
 *         Hao Yin <haoyin@uw.edu>
 */

#pragma once
#include "ns3/ns3-ai-dl.h"
#include "ns3/ff-mac-common.h"

namespace ns3 {
#define MAX_RBG_NUM 32

/**
 * \brief The feature of bsr.
 * 
 * The feature of DL training (in this example, feature of bsr)
 * shared between ns-3 and python with the same shared memory
 * using the ns3-ai model.
 */
struct BsrFeature
{
  uint16_t org_bsr;                  ///< original bsr ;
  uint8_t rbgNum;                 ///< resource block group number
  uint8_t nLayers;                ///< number of layers
  uint8_t sbCqi[MAX_RBG_NUM][2];  ///< sub band cqi
};

/**
 * \brief The prediction of bsr.
 * 
 * The prediction of DL training (in this example, prediction of bsr)
 * calculated by python and put back to ns-3 with the shared memory.
 */
struct BsrPredicted
{
  uint16_t new_bsr;
  uint8_t new_sbCqi[MAX_RBG_NUM][2];
};

/**
 * \brief The target of rnti.
 * 
 * The target of DL training (in this example, target of rnti)
 */
struct BsrTarget
{
  uint8_t target;
};

/**
 * \brief A class to predict BSR(Buffer status report).
 *
 * This class shared memory with python by the same id.
 * It set data through member function 'Set[xxx]()', 
 * and put them into the shared memory, using python to calculate,
 * and got prediction through member function 'Get[xxx]()'.
 */
class BSRDL : public Ns3AIDL<BsrFeature, BsrPredicted, BsrTarget>
{
public:
  BSRDL (void) = delete;
  BSRDL (uint16_t id);
  void SetBsr (uint16_t org_bsr);
  uint16_t GetBsr (void);
  void SetSbCQI (SbMeasResult_s cqi, uint32_t nLayers);
  void GetSbCQI (SbMeasResult_s &cqi);
  void SetTarget (uint8_t tar);
  uint8_t GetTarget (void);
};

} // namespace ns3
