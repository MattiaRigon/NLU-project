import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False,batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

class LSTM(nn.Module):
  def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,emb_dropout=0.1, n_layers=1):
    super(LSTM,self).__init__()
    self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
    self.dropout1 = nn.Dropout(p=emb_dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False,batch_first=True)
    self.pad_token = pad_index
    self.dropout2 = nn.Dropout(p=out_dropout)
    self.output = nn.Linear(hidden_size, output_size)

  def forward(self, input_sequence):
    emb = self.embedding(input_sequence)
    drp1 = self.dropout1(emb)
    lstm_out, _  = self.lstm(drp1)
    drp2 = self.dropout2(lstm_out)
    output = self.output(drp2).permute(0,2,1)
    return output
