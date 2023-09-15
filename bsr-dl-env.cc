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

#include "bsr-dl-env.h"

/**
 * \brief Link the shared memory with the id and set the operation lock
 * 
 * \param[in] id  shared memory id, should be the same in python and ns-3
 */
namespace ns3 {
BSRDL::BSRDL (uint16_t id) : Ns3AIDL<BsrFeature, BsrPredicted, BsrTarget> (id)
{
  SetCond (2, 0);
}

/**
 * \brief Set the value of bsr.
 * 
 * \param[in] bsr  the value of bsr to be set
 */
void
BSRDL::SetBsr (uint16_t bsr)
{
  auto feature = FeatureSetterCond ();    ///< get pointer to modify feature
  feature->org_bsr = bsr;
  SetCompleted ();                        ///< modification completed
}

/**
 * \brief Get the predictive value of bsr.
 * 
 * \returns the predictive value of bsr
 */
uint16_t
BSRDL::GetBsr (void)
{
  auto pred = PredictedGetterCond ();     ///< get predicted pointer for reading
  uint16_t ret = pred->new_bsr;
  GetCompleted ();                        ///< read completed
  return ret;
}

/**
 * \brief Set the value of sbCqi, rbgNum and nLayers.
 * 
 * \param[in] cqi  the means result of sbcqi
 * 
 * \param[in] nLayers  the value of nLayers to be set
 */
void
BSRDL::SetSbCQI (SbMeasResult_s cqi, uint32_t nLayers)
{
  auto feature = FeatureSetterCond ();    ///< get pointer to modify feature
  uint32_t rbgNum = cqi.m_higherLayerSelected.size ();
  feature->rbgNum = rbgNum;
  feature->nLayers = nLayers;
  for (uint32_t i = 0; i < rbgNum; ++i)
    {
      for (uint32_t j = 0; j < nLayers; ++j)
        {
          feature->sbCqi[i][j] = cqi.m_higherLayerSelected.at (i).m_sbCqi.at (j);
        }
    }
  SetCompleted ();                        ///< modification completed
}

/**
 * \brief Get the predictive value of sbcqi.
 * 
 * \param[out] cqi  the address of cqi
 */
void
BSRDL::GetSbCQI (SbMeasResult_s &cqi)
{
  auto feature = FeatureGetterCond ();    ///< get feature pointer for reading
  auto pred = PredictedGetterCond ();     ///< get predicted pointer for reading
  uint32_t rbgNum = feature->rbgNum;
  uint32_t nLayers = feature->nLayers;

  for (uint32_t i = 0; i < rbgNum; ++i)
    {
      for (uint32_t j = 0; j < nLayers; ++j)
        {
          cqi.m_higherLayerSelected.at (i).m_sbCqi.at (j) = pred->new_sbCqi[i][j];
        }
    }
  GetCompleted ();                        ///< read completed
}

/**
 * \brief Set the target rnti.
 * 
 * \param[in] tar  the value of target rnti to be set
 */
void BSRDL::SetTarget (uint8_t tar)
{
  auto target = TargetSetterCond ();      ///< get pointer to modify target
  target->target = tar;
  SetCompleted ();                        ///< modification completed
}

/**
 * \brief Get the target.
 * 
 * \returns the value of target
 */
uint8_t BSRDL::GetTarget (void)
{
  auto tar = TargetGetterCond ();         ///< get target pointer for reading
  uint8_t ret = tar->target;
  GetCompleted ();                        ///< read completed
  return ret;
}
} // namespace ns3
