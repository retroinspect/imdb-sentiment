import torch.nn as nn
import torch

class RNN(nn.Module):
  """
  __init__: init embeddings with random tensor if pretrained_embedding is None
  """
  def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None, useGPU=True):
    super(RNN, self).__init__()
    if (pretrained_embedding is not None):
      self.embedding_layer = pretrained_embedding
    else:
      self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=1)

    self.hidden_size = hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.useGPU = useGPU

  """
  forward_step: update hidden layer only if idx < update_bound
  """
  def forward_step(self, idx, hidden, update_bound=None):
    input = self.embedding_layer(idx)
    input = input.squeeze(1)
    old_hidden = hidden
    combined = torch.cat([input, hidden], dim=1) # combined.shape : batch_size, (input_size + hidden_size)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    if (update_bound != None):
      hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
    return output, hidden

  """
  forward: feed text_tensor into model word by word
  """
  def forward(self, text_tensor, update_bounds):
    hidden = self._init_hidden(batch_size=text_tensor.size(1))
    for i in range(text_tensor.size(0)):
      output, hidden = self.forward_step(text_tensor[i], hidden, update_bounds[i])
    return output

  """ 
  _init_hidden
  TODO compare torch.rand vs torch.zeros
  """
  def _init_hidden(self, batch_size):
    hidden = torch.rand(batch_size, self.hidden_size)
    if self.useGPU:
      hidden = hidden.cuda()
    return hidden

"""
GRU model with built-in nn.GRU module 
TODO replace with MyGRUCell
"""
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None, n_classes=2, useGPU=True, dropout_p=0.2):
        super(GRU, self).__init__()
        if (pretrained_embedding is not None):
          self.embedding_layer = pretrained_embedding
        else:
          # init embeddings with random tensor
          self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=1)

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, n_classes)
        self.useGPU = useGPU

    def forward(self, text_tensor, update_bounds):
      input = self.embedding_layer(text_tensor)
      input = input.squeeze(2) # input.shape: sequence length * batch_size * embedding_size
      hidden = torch.zeros(1, text_tensor.size(1), self.hidden_size) # hidden.shape: 1 * batch_size * hidden_size
      if self.useGPU:
        hidden = hidden.cuda()
      output, _ = self.gru(input, hidden)
      hidden_t = output[-1, :, :] # hidden_t.shape: batch_size * hidden_size
      self.dropout(hidden_t)
      final = self.out(hidden_t) # final.shape: batch_size * n_classes
      return final 

"""
Handmade GRU cell 
TODO debug
TODO implement biderectional GRU
"""
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
      super(MyGRUCell, self).__init__()

      self.input_size = input_size
      self.hidden_size = hidden_size

      self.reset_weight_input = nn.Linear(input_size, hidden_size)
      self.reset_weight_hidden = nn.Linear(hidden_size, hidden_size)
      self.sigmoid = nn.Sigmoid()
      
      self.weight_input = nn.Linear(input_size, hidden_size)
      self.weight_hidden = nn.Linear(hidden_size, hidden_size)
      self.tanh = nn.Tanh()

      self.update_weight_input = nn.Linear(input_size, hidden_size)
      self.update_weight_hidden = nn.Linear(hidden_size, hidden_size)

    def forward_step(self, input, hidden, update_bound=None):            
      old_hidden = hidden
      reset_signal = self.sigmoid(self.reset_weight_input(input) + self.reset_weight_hidden(hidden))
      new_mem = self.tanh(reset_signal * self.reset_weight_hidden(hidden) + self.reset_weight_input(input))      
      update_signal = self.sigmoid(self.update_weight_input(input) + self.update_weight_hidden(hidden))
      hidden = new_mem * (1-update_signal) + update_signal * hidden
      if (update_bound != None):
        hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
      return hidden

    def forward(self, inputs, update_bounds):
      hidden = torch.rand(inputs.size(1), self.hidden_size)
      if self.useGPU:
        hidden = hidden.cuda()      
      output = []
      for i in range(inputs.size(0)):
        hidden = self.forward_step(inputs[i], hidden, update_bounds[i])
        output.append(hidden)
      return output, hidden