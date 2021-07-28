import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import build_vocab
from load_data import IMDBDataset, split_data, MyCollator
from model import RNN, GRU
from train import train
from validate import validate

"""
determine whether to use GPU / CPU
"""
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
vocabulary = build_vocab(train_set)
vocab_size = len(vocabulary)
print(f'vocab size: {vocab_size}')
collate_fn = MyCollator(vocabulary)
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

"""
model configuration
- pretrained_embedding: None, TODO GloVe
- model: RNN, GRU
- optimizer: SGD, Adam, etc
"""
model = GRU(embedding_size, n_hidden, n_classes, vocab_size)
if useGPU:
  model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
train & validate
"""
for iter in range(n_iters):
  train_loss, train_accuracy = train(model, optimizer, train_loader)
  valid_loss, valid_accuracy, valid_known_rate = validate(model, valid_loader)

  print(f"[{iter+1}/{n_iters}] Train Loss : {train_loss:.4f} Train Acc : {train_accuracy:.2f} \
  Valid Loss : {valid_loss:.4f} Valid Acc : {valid_accuracy:.2f} Valid Word in vocabulary : {valid_known_rate:.2f}")

torch.save(model, 'imdb-rnn-classification.pt')

"""
TODO test
"""