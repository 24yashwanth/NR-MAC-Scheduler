from py_interface import *
from ctypes import *
import gc
from numpy import array

import torch
import torch.nn as nn
import math
from torch.serialization import load

import random


filename = './Extra Files/model/Transformers_BSR_model.pth'

n_steps = 10

# Hyperparameters
input_dim = n_steps  # Input dimension (sequence length)
num_blocks = 4  # Number of transformer blocks
d_model = 128    # Dimension of the model
num_heads = 4   # Number of attention heads
ff_dim = 32     # Dimension of the feedforward network
dropout_rate = 0.01  # Dropout rate
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

n_features = 1


x_input = []
for i in range(100):
    temp_bsr_queues = []
    for i in range(n_steps):
        temp_bsr_queues.append(random.randint(1,10000))
    x_input = array(temp_bsr_queues)
    x_input = x_input.reshape(n_features,n_steps)
    with torch.no_grad():
        input = torch.tensor(x_input, dtype=torch.float32)
    yhat = model(input)
    pred_val = int(yhat)
    print(input, " -> ", pred_val, end = "\n")
    