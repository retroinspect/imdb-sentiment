import torch.nn as nn
import torch

# TODO GloVe도 설정에 따라 이용할 수 있게 하려면 어떻게 해야할까?
# TODO Bidirectional RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None, useGPU=True):
        super(RNN, self).__init__()
        # embedding dictionary 인자로 넣고 opimize

        if (pretrained_embedding is not None):
          self.embedding_layer = pretrained_embedding
          self.pretrained = True
        else:
          # init embeddings with random tensor
          self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=1)
          self.pretrained = False

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.useGPU = useGPU


    def forward_step(self, idx, hidden, update_bound=None):
      # TODO idx=0(모르는 단어) 처리 시 embedding_layer에 해당 단어에 대한 임베딩 추가

      input = self.embedding_layer(idx)
      input = input.squeeze(1)
            
      old_hidden = hidden
      combined = torch.cat([input, hidden], dim=1) # combined.shape : batch_size, (input_size + hidden_size)
      hidden = self.i2h(combined)
      output = self.i2o(combined)

      # update_bound 미만의 index에 대해서만 hidden update
      # 이상의 index에 대해서는 old_hidden 사용
      if (update_bound != None):
        hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
      # update_bound = 4 이면 hidden[3, :] + old_hidden[4:, :] 

      return output, hidden

    def forward(self, text_tensor, update_bounds):

    # 초기화: 리뷰 배치에 대해 각각의 리뷰의 정보를 저장할 hidden state 초기화하여 반환
    # TODO torch.rand vs torch.zeros
      hidden = torch.zeros(text_tensor.size(1), self.hidden_size)
    
      if self.useGPU:
        hidden = hidden.cuda()
    # feed into model word by word
      for i in range(text_tensor.size(0)):
        output, hidden = self.forward_step(text_tensor[i], hidden, update_bounds[i])
      return output


# use nn.GRU
# TODO use MyGRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None, n_classes=2, useGPU=True, dropout_p=0.2):
        super(GRU, self).__init__()
        if (pretrained_embedding is not None):
          self.embedding_layer = pretrained_embedding
          self.pretrained = True
        else:
          # init embeddings with random tensor
          self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=1)
          self.pretrained = False

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, n_classes)
        self.useGPU = useGPU

    def forward(self, text_tensor, update_bounds):
      input = self.embedding_layer(text_tensor)
      input = input.squeeze(2)

      # print(input.shape) # sequence length * batch_size * embedding_size
      # assert input.dim() == 3, f"dimension of input should be 3, but found {input.dim()}" 

      hidden = torch.zeros(1, text_tensor.size(1), self.hidden_size) # 1 * batch_size * hidden_size
      if self.useGPU:
        hidden = hidden.cuda()

      output, _ = self.gru(input, hidden)
      hidden_t = output[-1, :, :] # batch_size * hidden_size

      self.dropout(hidden_t)
      return self.out(hidden_t) # batch_size * n_classes

# TODO implement GRU
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None, useGPU=True):
        super(MyGRU, self).__init__()
        # embedding dictionary 인자로 넣고 opimize

        if (pretrained_embedding is not None):
          self.embedding_layer = pretrained_embedding
          self.pretrained = True
        else:
          # init embeddings with random tensor
          self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=1)
          self.pretrained = False

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.useGPU = useGPU


    def forward_step(self, idx, hidden, update_bound=None):
      # TODO idx=0(모르는 단어) 처리 시 embedding_layer에 해당 단어에 대한 임베딩 추가

      input = self.embedding_layer(idx)
      input = input.squeeze(1)
            
      old_hidden = hidden
      combined = torch.cat([input, hidden], dim=1) # combined.shape : batch_size, (input_size + hidden_size)
      hidden = self.i2h(combined)
      output = self.i2o(combined)

      # update_bound 미만의 index에 대해서만 hidden update
      # 이상의 index에 대해서는 old_hidden 사용
      if (update_bound != None):
        hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
      # update_bound = 4 이면 hidden[3, :] + old_hidden[4:, :] 

      return output, hidden

    def forward(self, text_tensor, update_bounds):

    # 초기화: 리뷰 배치에 대해 각각의 리뷰의 정보를 저장할 hidden state 초기화하여 반환
    # TODO torch.rand vs torch.zeros
      hidden = torch.zeros(text_tensor.size(1), self.hidden_size)
    
      if self.useGPU:
        hidden = hidden.cuda()
    # feed into model word by word
      for i in range(text_tensor.size(0)):
        output, hidden = self.forward_step(text_tensor[i], hidden, update_bounds[i])
      return output
