import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, useGPU):
        super(RNN, self).__init__()
        # embedding dictionary 인자로 넣고 opimize

        # init embeddings with random tensor
        self.embedding_size = input_size
        self.embeddings = torch.zeros(vocab_size, input_size)
        for idx in range(1, self.embeddings.size(0)):
          self.embeddings[idx] = torch.rand(self.embedding_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.useGPU = useGPU

        # dimension 맞추기
    def forward(self, idx, hidden, update_bound=None):
      # TODO batch에 대해 embedding 어떻게 조회???
      # def getEmbedding(idx):
      #   if (idx > 0): # 에러 발생
         #     return self.embeddings[idx]
          # expand vocab
        # new_embedding = torch.rand(self.embedding_size).unsqueeze(0)
        # self.embeddings = torch.cat([self.embedding, new_embedding])
        # new_vocab_size = self.embeddings.size(0)
        # return self.embeddings[new_vocab_size-1]

      input = self.embeddings[idx].squeeze(1)
      if self.useGPU:
        input = input.cuda()
      
      old_hidden = hidden
      combined = torch.cat([input, hidden], dim=1) # combined.shape : batch_size, (input_size + hidden_size)
      hidden = self.i2h(combined)
      output = self.i2o(combined)
      output = self.softmax(output)

      # update_bound 미만의 index에 대해서만 hidden update
      # 이상의 index에 대해서는 old_hidden 사용
      if (update_bound != None):
        hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
      # update_bound = 4 이면 hidden[3, :] + old_hidden[4:, :] 

      return output, hidden

    def initHidden(self, sample_batched_size):
      return torch.zeros(sample_batched_size, self.hidden_size)      

