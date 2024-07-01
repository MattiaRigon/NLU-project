import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import DEVICE

class VariationalDropout(nn.Module):
  def __init__(self,dropout_rate):
    super(VariationalDropout, self).__init__()
    self.dropout_rate = dropout_rate

  def __mask_data(self,data):

    """
      data: [batch_size, length_of_sequence, layers_dimentions ]      
    """

    binomial = torch.distributions.binomial.Binomial(probs=1-self.dropout_rate)
    m = binomial.sample((data.shape[0],1,data.shape[2])).to(DEVICE)
    mask = m.expand(data.shape[0],data.shape[1],data.shape[2])
    masked_data = mask * data * (1/(1-self.dropout_rate))
    return masked_data


  def forward(self, x):

    """
      x: [batch_size, length_of_sequence, layers_dimentions ]      
    """

    if not self.training:
      return x

    data = x
    pack_length = 1  
    is_packed = False

    if isinstance(x, nn.utils.rnn.PackedSequence):
      is_packed = True
      data, pack_length = pad_packed_sequence(x, batch_first=True)

    data = self.__mask_data(data)

    if is_packed:
      return pack_padded_sequence(data,pack_length)
    else :
      return data



class LSTM(nn.Module):
  def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
    """
    Initialize the LSTM model.

    Args:
      emb_size (int): The size of the word embeddings.
      hidden_size (int): The size of the hidden state of the LSTM.
      output_size (int): The size of the output.
      pad_index (int, optional): The index of the padding token. Defaults to 0.
      out_dropout (float, optional): The dropout rate for the output layer. Defaults to 0.1.
      emb_dropout (float, optional): The dropout rate for the word embeddings. Defaults to 0.1.
      n_layers (int, optional): The number of LSTM layers. Defaults to 1.
    """
    super(LSTM, self).__init__()
    self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
    self.dropout1 = nn.Dropout(p=emb_dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
    self.pad_token = pad_index
    self.dropout2 = nn.Dropout(p=out_dropout)
    self.output = nn.Linear(hidden_size, output_size)
    self.variationalDropout = VariationalDropout(out_dropout)
    assert emb_size == hidden_size # ensure that the embedding size is equal to the hidden size, in order to apply weight tying
    self.output.weight = self.embedding.weight  # Apply weight tying

  def forward(self, input_sequence):
    """
    Forward pass of the model.

    Args:
      input_sequence (torch.Tensor): Input sequence tensor.

    Returns:
      torch.Tensor: Output tensor after passing through the model.
    """
    emb = self.embedding(input_sequence)
    var_drop1 = self.variationalDropout(emb)
    lstm_out, _ = self.lstm(var_drop1)
    var_drop2 = self.variationalDropout(lstm_out)
    output = self.output(var_drop2).permute(0, 2, 1)
    return output
