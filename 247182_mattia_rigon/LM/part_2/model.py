import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class VariationalDropout(nn.Module):
  def __init__(self,dropout_rate):
    super(VariationalDropout, self).__init__()
    self.dropout_rate = dropout_rate

  def forward(self, x):

    if not self.training:
      return x
    
    if isinstance(x, nn.utils.rnn.PackedSequence):
      mask = torch.bernoulli((1-self.dropout_rate) * torch.ones(x.data.size(0), x.data.size(1), x.data.size(2))).to(x.data.device)
      masked_data = mask.unsqueeze(2) * x.data * (1.0/(1-self.dropout_rate))
      return nn.utils.rnn.PackedSequence(masked_data, x.batch_sizes)
    else:
      mask = torch.bernoulli((1-self.dropout_rate) * torch.ones(x.size(0), x.size(1), x.size(2))).to(x.device)
      return mask.unsqueeze(2) * x * (1.0/(1-self.dropout_rate))

class LSTM(nn.Module):
  def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,emb_dropout=0.1, n_layers=1):
    super(LSTM,self).__init__()
    self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
    self.dropout1 = nn.Dropout(p=emb_dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False,batch_first=True)
    self.pad_token = pad_index
    self.dropout2 = nn.Dropout(p=out_dropout)
    self.output = nn.Linear(hidden_size, output_size)
    assert emb_size == hidden_size
    self.output.weight = self.embedding.weight # Apply weight tying

  def forward(self, input_sequence):
    emb = self.embedding(input_sequence)
    drp1 = self.dropout1(emb)
    lstm_out, _  = self.lstm(drp1)
    drp2 = self.dropout2(lstm_out)
    output = self.output(drp2).permute(0,2,1)
    return output
