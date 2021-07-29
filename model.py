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
    def __init__(self, input_size, hidden_size, vocab_size, pretrained_embedding=None, n_classes=2, useGPU=True, dropout_p=0.2):
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
        self.sigmoid = nn.Sigmoid()

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
      final = self.sigmoid(final)
      return final 

"""
Wikidocs: model from https://wikidocs.net/60691
- accuracy from the article: 87%
- actual accuracy: 52%
"""
class Wikidocs(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_vocab, n_classes=2, dropout_p=0.2):
        super(Wikidocs, self).__init__()
        self.n_layers = 1
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x, update_bounds):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(1)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (시퀀스 길이, 배치 크기, 은닉 상태의 크기)
        h_t = x[-1:, :, :] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        # print(logit.shape)
        return logit.squeeze(0)

    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

"""
SentimentRNN: model from https://www.kaggle.com/arunmohan003/sentiment-analysis-using-lstm-pytorch

"""
class SentimentRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim=1, no_layers=2, drop_prob=0.5):
    super(SentimentRNN,self).__init__()

    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    self.no_layers = no_layers
    self.vocab_size = vocab_size

    # embedding and LSTM layers
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True) 
    self.dropout = nn.Dropout(0.3)

    # linear and sigmoid layer
    self.fc = nn.Linear(self.hidden_dim, output_dim)
    self.sig = nn.Sigmoid()
      
  def forward(self, x, hidden):
    # print(x.shape)
    # x = x.reshape(x.size(1), x.size(0))
    batch_size = x.size(0)

    # embeddings and lstm_out
    embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
    
    # dropout and fully connected layer
    out = self.dropout(lstm_out)
    out = self.fc(out)
    
    # sigmoid function
    sig_out = self.sig(out)
    
    # reshape to be batch_size first
    sig_out = sig_out.view(batch_size, -1)

    sig_out = sig_out[:, -1] # get last batch of labels
    
    # return last sigmoid output and hidden state
    return sig_out, hidden
      
      
      
  def init_hidden(self, batch_size, device):
    ''' Initializes hidden state '''
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
    c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
    hidden = (h0,c0)
    return hidden




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