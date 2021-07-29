import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import build_vocab
from load_data import IMDBDataset, split_data, MyCollator
from model import RNN, GRU, Wikidocs, SentimentRNN
from train import train
from validate import validate

"""
determine whether to use GPU / CPU
"""
# useGPU = False
useGPU = torch.cuda.is_available()
device = torch.device("cuda" if useGPU else "cpu")
print(f"using {device}")

"""
hyperparameters
"""
batch_size=4
learning_rate=0.005
n_iters=10
embedding_size=200
n_classes = 2 # positive, negative
n_hidden = 256

"""
load & split data
"""
dataset = IMDBDataset(datapath='imdb_binary_sent.csv')
train_set, valid_set, test_set = split_data(dataset)
print(f'train set size: {len(train_set)}')
print(f'validation set size: {len(valid_set)}')
print(f'test set size: {len(test_set)}')


""" 
build vocabulary & dataloader
"""
try:
  vocab = torch.load('vocab.pth')
except:
  vocab = build_vocab(train_set)
  torch.save(vocab, 'vocab.pth')

vocab_size = len(vocab) + 1
print(f'Length of vocabulary is {len(vocab)}')

collate_fn = MyCollator(vocab)
valid_loader = DataLoader(valid_set, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)

"""
model configuration
- pretrained_embedding: None, TODO GloVe
- model: RNN, GRU
- optimizer: SGD, Adam, etc
"""
model = SentimentRNN(embedding_size, n_hidden, vocab_size)
# model = GRU(embedding_size, n_hidden, vocab_size)

if useGPU:
  model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

from tqdm import tqdm
import numpy as np

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

clip = 5
epochs = 5 

if useGPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

for epoch in range(epochs):
  train_losses = []
  train_acc = 0.0
  model.train()
  # initialize hidden state 
  for batch_idx, sample_batched in enumerate(tqdm(train_loader, desc="training...")):

    inputs, labels = sample_batched
    inputs, labels = inputs.to(device), labels.to(device)   
    h = model.init_hidden(inputs.size(0), device)
    
    model.zero_grad()
    output,h = model(inputs,h)
    
    # calculate the loss and perform backprop
    loss = criterion(output.squeeze(), labels.float())
    loss.backward()
    train_losses.append(loss.item())
    # calculating accuracy
    accuracy = acc(output,labels)
    train_acc += accuracy
    #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
        
  val_losses = []
  val_acc = 0.0
  model.eval()
  for batch_idx, sample_batched in enumerate(tqdm(valid_loader, desc="validating...")):
    inputs, labels = sample_batched
    inputs, labels = inputs.to(device), labels.to(device)

    val_h = model.init_hidden(inputs.size(0), device)
    output, val_h = model(inputs, val_h)
    val_loss = criterion(output.squeeze(), labels.float())

    val_losses.append(val_loss.item())
    
    accuracy = acc(output,labels)
    val_acc += accuracy
          
  epoch_train_loss = np.mean(train_losses)
  epoch_val_loss = np.mean(val_losses)
  epoch_train_acc = train_acc/len(train_loader.dataset)
  epoch_val_acc = val_acc/len(valid_loader.dataset)
  print(f'Epoch {epoch+1}') 
  print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
  print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
  print(25*'==')


# """
# train & validate
# """
# for iter in range(n_iters):
#   train_loss, train_accuracy = train(model, optimizer, train_loader, useGPU)
#   valid_loss, valid_accuracy, valid_known_rate = validate(model, valid_loader, useGPU)

#   print(f"[{iter+1}/{n_iters}] Train Loss : {train_loss:.4f} Train Acc : {train_accuracy:.2f} \
#   Valid Loss : {valid_loss:.4f} Valid Acc : {valid_accuracy:.2f} Valid Word in vocabulary : {valid_known_rate:.2f}")

# torch.save(model, 'imdb-rnn-classification.pt')

# """
# TODO test
# """