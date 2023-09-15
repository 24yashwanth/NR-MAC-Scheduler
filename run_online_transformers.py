# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# Copyright (c) 2019 Huazhong University of Science and Technology, Dian Group
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Xiaojun Guo <guoxj@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>

from py_interface import *
from ctypes import *
import gc
from numpy import array

import torch
import torch.nn as nn
import math
from torch.serialization import load

# =================================================

# n_steps is the range of bsr values taken into consideration for predicting the next bsr value
n_steps = 10


# Model is loaded here for predictions 
filename = './Extra Files/model/Transformers_BSR_model.pth'
# Hyperparameters
input_dim = n_steps  # Input dimension (sequence length)
num_blocks = 2  # Number of transformer blocks
d_model = 64    # Dimension of the model
num_heads = 4   # Number of attention heads
ff_dim = 32     # Dimension of the feedforward network
dropout_rate = 0.1  # Dropout rate
output_dim = 1  # Output dimension (for regression)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_positional_encoding(max_len, d_model))

    def _get_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # Calculate the positional encoding for the input sequence x
        pe = self.pe[:, :x.size(0)]
        x = x + pe  # Broadcast pe to match the dimensions of x
        return x
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, d_model, ff_dim, dropout_rate, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff_dim, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Aggregate sequence information
        x = self.fc(x)
        return x

model = TransformerModel(input_dim=input_dim, nhead=num_heads, d_model=d_model, num_layers=num_blocks, ff_dim=ff_dim, dropout_rate=dropout_rate, output_dim=output_dim)
model.load_state_dict(load(filename))
model.eval()
# model = keras.models.load_model("my_model")

# delta for prediction
# delta = int(sys.argv[1])

MAX_RBG_NUM = 32

# classes are made here to replicate the c++ side to facilitate easy data transfer

class BsrFeature(Structure):
    _pack_ = 1
    _fields_ = [('org_bsr', c_uint16), ('rbgNum', c_uint8), ('nLayers', c_uint8),
                ('sbCqi', (c_uint8 * MAX_RBG_NUM) * 2)]


class BsrPredicted(Structure):
    _pack_ = 1
    _fields_ = [('new_bsr', c_uint16),
                ('new_sbCqi', (c_uint8 * MAX_RBG_NUM) * 2)]


class BsrTarget(Structure):
    _pack_ = 1
    _fields_ = [('target', c_uint8)]


mempool_key = 1234          # memory pool key, arbitrary integer large than 1000
mem_size = 16384            # memory pool size in bytes (4096)
Init(mempool_key, mem_size) # Init shared memory pool

memblock_key = 1357         # memory block key, need to keep the same in the ns-3 script
dl = Ns3AIDL(memblock_key, BsrFeature, BsrPredicted, BsrTarget)     # Link the shared memory block with ns-3 script





# BSR, RNTI are used to store bsr and UE rnti values from shared memory

BSR = 0
RNTI=0
n_features = 1

# temp_bsr_queues list is made to hold the bsr values equal to n_steps so that we can run our prediction on each queue for different UEs

temp_bsr_queues = {}
bsr_queue = []
rnti_queue = []


// with and without prediction BSR values plot

exp = Experiment(mempool_key, mem_size, 'TRANSFORMERS', '../../')
exp.run(show_output=1)
try:
    while True:
        with dl as data:
            if dl.isFinish():
                break
            gc.collect()
            # # Get BSR and RNTI from the shared memory
            BSR = data.feat.org_bsr
            RNTI=data.tar.target
            bsr_queue.append(BSR)
            rnti_queue.append(RNTI)

            # ------------------------------------------
            
            # Here we check the RNTI and accordingly perform the required operations
            # predictions are done on each temp_bsr_queues of length n_steps to get the next value
            

            # Check if the RNTI key exists in the dictionary, if not, create a new list
            if RNTI not in temp_bsr_queues:
                temp_bsr_queues[RNTI] = []

            # Append BSR to the corresponding list
            temp_bsr_queues[RNTI].append(BSR)

            # Check if the length of the list is less than n_steps
            if len(temp_bsr_queues[RNTI]) < n_steps:
                data.pred.new_bsr = BSR

            # Check if the length of the list is equal to n_steps
            if len(temp_bsr_queues[RNTI]) == n_steps:
                x_input = array(temp_bsr_queues[RNTI])

                for item in range(len(x_input)):
                    x_input[item] = float(x_input[item])
                
                x_input = x_input.reshape(n_features,n_steps)
                with torch.no_grad():
                    input = torch.tensor(x_input, dtype=torch.float32)
                    yhat = model(input)
                pred_val = int(yhat)

                if pred_val < 0 or pred_val < BSR:
                    data.pred.new_bsr = BSR
                else:
                    data.pred.new_bsr = pred_val

                # Remove the first element from the list
                temp_bsr_queues[RNTI] = temp_bsr_queues[RNTI][1:]




    print(bsr_queue)
    print(rnti_queue)
    print(temp_bsr_queues)

except KeyboardInterrupt:
    print('Ctrl C')
finally:
    del exp
print('Finish')



# -------------------------------------------------------------------------------------------------------











# temp_bsr_queue_ue1 = []
# temp_bsr_queue_ue2 = []
# temp_bsr_queue_ue3 = []
# temp_bsr_queue_ue4 = []
# temp_bsr_queue_ue5 = []
# temp_bsr_queue_ue6 = []
# temp_bsr_queue_ue7 = []
# temp_bsr_queue_ue8 = []
# temp_bsr_queue_ue9 = []
# temp_bsr_queue_ue10 = []
# temp_bsr_queue_ue11 = []
# temp_bsr_queue_ue12 = []
# temp_bsr_queue_ue13 = []
# temp_bsr_queue_ue14 = []
# temp_bsr_queue_ue15 = []
# temp_bsr_queue_ue16 = []
# temp_bsr_queue_ue17 = []
# temp_bsr_queue_ue18 = []
# temp_bsr_queue_ue19 = []
# temp_bsr_queue_ue20 = []
# temp_bsr_queue_ue21 = []
# temp_bsr_queue_ue22 = []
# temp_bsr_queue_ue23 = []
# temp_bsr_queue_ue24 = []
# temp_bsr_queue_ue25 = []
# temp_bsr_queue_ue26 = []
# temp_bsr_queue_ue27 = []
# temp_bsr_queue_ue28 = []
# temp_bsr_queue_ue29 = []
# temp_bsr_queue_ue30 = []







            # if(RNTI==1):
            #     temp_bsr_queue_ue1.append(BSR) 
            #     if(len(temp_bsr_queue_ue1)<n_steps):
            #         data.pred.new_bsr = BSR 
            #     if(len(temp_bsr_queue_ue1)==n_steps):
            #         x_input=array(temp_bsr_queue_ue1)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR 
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val       
            #         temp_bsr_queue_ue1=temp_bsr_queue_ue1[1:]
            # elif(RNTI==2):
            #     temp_bsr_queue_ue2.append(BSR) 
            #     if(len(temp_bsr_queue_ue2)<n_steps):
            #         data.pred.new_bsr = BSR 
            #     if(len(temp_bsr_queue_ue2)==n_steps):
            #         x_input=array(temp_bsr_queue_ue2)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR 
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val       
            #         temp_bsr_queue_ue2=temp_bsr_queue_ue2[1:]
            # elif(RNTI==3):
            #     temp_bsr_queue_ue3.append(BSR) 
            #     if(len(temp_bsr_queue_ue3)<n_steps):
            #         data.pred.new_bsr = BSR 
            #     if(len(temp_bsr_queue_ue3)==n_steps):
            #         x_input=array(temp_bsr_queue_ue3)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR 
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val       
            #         temp_bsr_queue_ue3=temp_bsr_queue_ue3[1:]   
            # elif(RNTI==4):
            #     temp_bsr_queue_ue4.append(BSR) 
            #     if(len(temp_bsr_queue_ue4)<n_steps):
            #         data.pred.new_bsr = BSR 
            #     if(len(temp_bsr_queue_ue4)==n_steps):
            #         x_input=array(temp_bsr_queue_ue4)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR 
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val       
            #         temp_bsr_queue_ue4=temp_bsr_queue_ue4[1:]   
            # elif(RNTI==5):
            #     temp_bsr_queue_ue5.append(BSR) 
            #     if(len(temp_bsr_queue_ue5)<n_steps):
            #         data.pred.new_bsr = BSR 
            #     if(len(temp_bsr_queue_ue5)==n_steps):
            #         x_input=array(temp_bsr_queue_ue5)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR 
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val      
            #         temp_bsr_queue_ue5=temp_bsr_queue_ue5[1:]
            # elif(RNTI==6):
            #     temp_bsr_queue_ue6.append(BSR) 
            #     if(len(temp_bsr_queue_ue6)<n_steps):
            #         data.pred.new_bsr = BSR+300
            #     if(len(temp_bsr_queue_ue6)==n_steps):
            #         x_input=array(temp_bsr_queue_ue6)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val=int(yhat)
            #         # print(yhat)
            #         if(pred_val<0 or pred_val<BSR):
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = BSR+300
            #         else:
            #             # prediction_queue.append(pred_val)
            #             data.pred.new_bsr = pred_val+300      
            #         temp_bsr_queue_ue6=temp_bsr_queue_ue6[1:]   
            # # Continue for RNTI values 7 to 10
            # elif RNTI == 7:
            #     temp_bsr_queue_ue7.append(BSR) 
            #     if len(temp_bsr_queue_ue7) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue7) == n_steps:
            #         x_input = array(temp_bsr_queue_ue7)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue7 = temp_bsr_queue_ue7[1:]
            # # Continue for RNTI values 8 to 10
            # elif RNTI == 8:
            #     temp_bsr_queue_ue8.append(BSR) 
            #     if len(temp_bsr_queue_ue8) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue8) == n_steps:
            #         x_input = array(temp_bsr_queue_ue8)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue8 = temp_bsr_queue_ue8[1:]
            # # Continue for RNTI values 9 to 10
            # elif RNTI == 9:
            #     temp_bsr_queue_ue9.append(BSR) 
            #     if len(temp_bsr_queue_ue9) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue9) == n_steps:
            #         x_input = array(temp_bsr_queue_ue9)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue9 = temp_bsr_queue_ue9[1:]
            # # Continue for RNTI value 10
            # elif RNTI == 10:
            #     temp_bsr_queue_ue10.append(BSR) 
            #     if len(temp_bsr_queue_ue10) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue10) == n_steps:
            #         x_input = array(temp_bsr_queue_ue10)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue10 = temp_bsr_queue_ue10[1:]
            # # Continue for RNTI values 11 to 30
            # elif RNTI == 11:
            #     temp_bsr_queue_ue11.append(BSR)
            #     if len(temp_bsr_queue_ue11) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue11) == n_steps:
            #         x_input = array(temp_bsr_queue_ue11)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue11 = temp_bsr_queue_ue11[1:]
            # # Continue the pattern for RNTI values 12 to 30
            # elif RNTI == 12:
            #     temp_bsr_queue_ue12.append(BSR)
            #     if len(temp_bsr_queue_ue12) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue12) == n_steps:
            #         x_input = array(temp_bsr_queue_ue12)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue12 = temp_bsr_queue_ue12[1:]
            # # Continue the pattern for RNTI values 13 to 30
            # elif RNTI == 13:
            #     temp_bsr_queue_ue13.append(BSR)
            #     if len(temp_bsr_queue_ue13) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue13) == n_steps:
            #         x_input = array(temp_bsr_queue_ue13)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue13 = temp_bsr_queue_ue13[1:]
            # # Continue the pattern for RNTI values 14 to 30
            # elif RNTI == 14:
            #     temp_bsr_queue_ue14.append(BSR)
            #     if len(temp_bsr_queue_ue14) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue14) == n_steps:
            #         x_input = array(temp_bsr_queue_ue14)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue14 = temp_bsr_queue_ue14[1:]
            # # Continue the pattern for RNTI values 15 to 30
            # elif RNTI == 15:
            #     temp_bsr_queue_ue15.append(BSR)
            #     if len(temp_bsr_queue_ue15) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue15) == n_steps:
            #         x_input = array(temp_bsr_queue_ue15)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue15 = temp_bsr_queue_ue15[1:]
            # # Continue the pattern for RNTI values 16 to 30
            # elif RNTI == 16:
            #     temp_bsr_queue_ue16.append(BSR)
            #     if len(temp_bsr_queue_ue16) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue16) == n_steps:
            #         x_input = array(temp_bsr_queue_ue16)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue16 = temp_bsr_queue_ue16[1:]
            # # Continue the pattern for RNTI values 17 to 30
            # elif RNTI == 17:
            #     temp_bsr_queue_ue17.append(BSR)
            #     if len(temp_bsr_queue_ue17) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue17) == n_steps:
            #         x_input = array(temp_bsr_queue_ue17)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue17 = temp_bsr_queue_ue17[1:]
            # # Continue the pattern for RNTI values 18 to 30
            # elif RNTI == 18:
            #     temp_bsr_queue_ue18.append(BSR)
            #     if len(temp_bsr_queue_ue18) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue18) == n_steps:
            #         x_input = array(temp_bsr_queue_ue18)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue18 = temp_bsr_queue_ue18[1:]
            # # Continue the pattern for RNTI values 19 to 30
            # elif RNTI == 19:
            #     temp_bsr_queue_ue19.append(BSR)
            #     if len(temp_bsr_queue_ue19) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue19) == n_steps:
            #         x_input = array(temp_bsr_queue_ue19)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue19 = temp_bsr_queue_ue19[1:]
            # # Continue the pattern for RNTI values 20 to 30
            # elif RNTI == 20:
            #     temp_bsr_queue_ue20.append(BSR)
            #     if len(temp_bsr_queue_ue20) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue20) == n_steps:
            #         x_input = array(temp_bsr_queue_ue20)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue20 = temp_bsr_queue_ue20[1:]
            # # Continue the pattern for RNTI values 21 to 30
            # elif RNTI == 21:
            #     temp_bsr_queue_ue21.append(BSR)
            #     if len(temp_bsr_queue_ue21) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue21) == n_steps:
            #         x_input = array(temp_bsr_queue_ue21)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue21 = temp_bsr_queue_ue21[1:]
            # # Continue the pattern for RNTI values 22 to 30
            # elif RNTI == 22:
            #     temp_bsr_queue_ue22.append(BSR)
            #     if len(temp_bsr_queue_ue22) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue22) == n_steps:
            #         x_input = array(temp_bsr_queue_ue22)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue22 = temp_bsr_queue_ue22[1:]
            # # Continue the pattern for RNTI values 23 to 30
            # elif RNTI == 23:
            #     temp_bsr_queue_ue23.append(BSR)
            #     if len(temp_bsr_queue_ue23) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue23) == n_steps:
            #         x_input = array(temp_bsr_queue_ue23)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue23 = temp_bsr_queue_ue23[1:]
            # # Continue the pattern for RNTI values 24 to 30
            # elif RNTI == 24:
            #     temp_bsr_queue_ue24.append(BSR)
            #     if len(temp_bsr_queue_ue24) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue24) == n_steps:
            #         x_input = array(temp_bsr_queue_ue24)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue24 = temp_bsr_queue_ue24[1:]
            # # Continue the pattern for RNTI values 25 to 30
            # elif RNTI == 25:
            #     temp_bsr_queue_ue25.append(BSR)
            #     if len(temp_bsr_queue_ue25) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue25) == n_steps:
            #         x_input = array(temp_bsr_queue_ue25)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue25 = temp_bsr_queue_ue25[1:]
            # # Continue the pattern for RNTI values 26 to 30
            # elif RNTI == 26:
            #     temp_bsr_queue_ue26.append(BSR)
            #     if len(temp_bsr_queue_ue26) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue26) == n_steps:
            #         x_input = array(temp_bsr_queue_ue26)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue26 = temp_bsr_queue_ue26[1:]
            # # Continue the pattern for RNTI values 27 to 30
            # elif RNTI == 27:
            #     temp_bsr_queue_ue27.append(BSR)
            #     if len(temp_bsr_queue_ue27) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue27) == n_steps:
            #         x_input = array(temp_bsr_queue_ue27)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue27 = temp_bsr_queue_ue27[1:]
            # # Continue the pattern for RNTI values 28 to 30
            # elif RNTI == 28:
            #     temp_bsr_queue_ue28.append(BSR)
            #     if len(temp_bsr_queue_ue28) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue28) == n_steps:
            #         x_input = array(temp_bsr_queue_ue28)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue28 = temp_bsr_queue_ue28[1:]
            # # Continue the pattern for RNTI values 29 to 30
            # elif RNTI == 29:
            #     temp_bsr_queue_ue29.append(BSR)
            #     if len(temp_bsr_queue_ue29) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue29) == n_steps:
            #         x_input = array(temp_bsr_queue_ue29)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue29 = temp_bsr_queue_ue29[1:]
            # # For RNTI value 30
            # elif RNTI == 30:
            #     temp_bsr_queue_ue30.append(BSR)
            #     if len(temp_bsr_queue_ue30) < n_steps:
            #         data.pred.new_bsr = BSR + 300
            #     if len(temp_bsr_queue_ue30) == n_steps:
            #         x_input = array(temp_bsr_queue_ue30)
            #         x_input = x_input.reshape((1, n_steps, n_features))
            #         yhat = model.predict(x_input, verbose=0)
            #         pred_val = int(yhat)
            #         if pred_val < 0 or pred_val < BSR:
            #             data.pred.new_bsr = BSR + 300
            #         else:
            #             data.pred.new_bsr = pred_val + 300
            #         temp_bsr_queue_ue30 = temp_bsr_queue_ue30[1:]





    # print(temp_bsr_queue_ue1) #Just to check ending
    # print(temp_bsr_queue_ue2)
    # print(temp_bsr_queue_ue3)
    # print(temp_bsr_queue_ue4)
    # print(temp_bsr_queue_ue5)
    # print(temp_bsr_queue_ue6)
    # print(temp_bsr_queue_ue7)
    # print(temp_bsr_queue_ue8)
    # print(temp_bsr_queue_ue9)
    # print(temp_bsr_queue_ue10)
    # print(temp_bsr_queue_ue11)
    # print(temp_bsr_queue_ue12)
    # print(temp_bsr_queue_ue13)
    # print(temp_bsr_queue_ue14)
    # print(temp_bsr_queue_ue15)
    # print(temp_bsr_queue_ue16)
    # print(temp_bsr_queue_ue17)
    # print(temp_bsr_queue_ue18)
    # print(temp_bsr_queue_ue19)
    # print(temp_bsr_queue_ue20)
    # print(temp_bsr_queue_ue21)
    # print(temp_bsr_queue_ue22)
    # print(temp_bsr_queue_ue23)
    # print(temp_bsr_queue_ue24)
    # print(temp_bsr_queue_ue25)
    # print(temp_bsr_queue_ue26)
    # print(temp_bsr_queue_ue27)
    # print(temp_bsr_queue_ue28)
    # print(temp_bsr_queue_ue29)
    # print(temp_bsr_queue_ue30)
            
