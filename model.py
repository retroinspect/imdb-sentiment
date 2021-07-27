import torch.nn as nn
import torch

# TODO GloVe도 설정에 따라 이용할 수 있게 하려면 어떻게 해야할까?
# TODO Bidirectional RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, pretrained_embedding=None):
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


    # TODO implement GRU
    # def gru_init(self, sample_batched_size):


    # def gru_step(self, idx, reset, update, hidden, prev_output):



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

    def forward(self, text_tensor, hidden, update_bounds):
    # feed into model word by word
      for i in range(text_tensor.size(0)):
        output, hidden = self.forward_step(text_tensor[i], hidden, update_bounds[i])
      return output, hidden


    def initHidden(self, sample_batched_size):
      return torch.zeros(sample_batched_size, self.hidden_size)      
