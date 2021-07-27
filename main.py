from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.nn as nn
import unicodedata
import string
import random
from utils import *
import dill
from model import RNN, GRU

# load pretrained word vector
# from torchtext.vocab import GloVe

# 랜덤 시드 고정
# SEED = 5
# random.seed(SEED)
# torch.manual_seed(SEED)

# use GPU / CPU
useGPU = torch.cuda.is_available()
print("cuda" if useGPU else "cpu")
device = torch.device("cuda" if useGPU else "cpu")

# hyperparameters
batch_size=128
learning_rate=0.005
n_iters=10
embedding_size=200

# load data
dataset = IMDBDataset()
train_set, valid_set, test_set = split_data(dataset)

print(f'train set size: {len(train_set)}')
print(f'test set size: {len(test_set)}')


# load vocab
while(True):
  try:
    print("loading vocab")
    vocabulary = torch.load('vocab.pth')
    break
  except:
    print("No vocab.pth found.. building new vocabulary")
    build_vocab(train_set)

vocab_size = len(vocabulary)
print(f'vocab size: {vocab_size}')

# TODO refactoring: dataset 만들기 -> vocab 만들기 -> loader 만들기 순서여야해서 리팩토링이 어려움
# 어떻게 하면 dependency 없앨 수 있을까

def textToLong(text):
  return torch.stack([tokenToLong(token) for token in text]).unsqueeze(1)

def tokenToLong(token):
  if not token in vocabulary:
    item = 0 # <unk>
  else:
    item = vocabulary[token]
  return torch.LongTensor([item])

# concat batch texts into tensor
def collate_fn(batch):
  # batch example: {label: 'positive', text: 'good movie'}
  labels, texts = [], []
  for sample in batch:
    label, txt = sample['label'], sample['text']
    texts.append(preprocess(txt))
    labels.append(label)
  
  text_lengths = torch.LongTensor([len(text) for text in texts])
  max_len = max(text_lengths)

  texts = torch.cat([textToLong(pad(text, max_len)) for text in texts], dim=1)
  labels = torch.LongTensor([label == 'positive' for label in labels])

  # texts.shape : max_len * batch_size * embedding_size
  # labels.shape : batch_size

  text_lengths, sorted_idx = text_lengths.sort(0, descending=True)
  texts = texts[:, sorted_idx]
  labels = labels[sorted_idx]
  
  # TODO update_bounds 좀 더 효율적으로 짜기
  update_bounds = torch.LongTensor([0] * max_len) # 한 번에 process하는 단어 개수 
  for text_len in text_lengths:
    tmp = torch.LongTensor([1] * text_len + [0] * (max_len - text_len))
    update_bounds += tmp;

  return (labels, texts, update_bounds)

train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

n_classes = 2 # pos, neg
n_hidden = 128

# TODO map GloVe into nn.Embedding
pretrained_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

# model = RNN(embedding_size, n_hidden, n_classes, vocab_size)
model = GRU(embedding_size, n_hidden, n_classes, vocab_size)

if useGPU:
  model.to(device)

# nn.BCEWithLogitsLoss includes softmax
# criterion = nn.BCEWithLogitsLoss()
# nn.BCEWithLogitsLoss vs F.cross_entropy

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import time
#  numpy arrays to save loss & accuracy from each epoch
train_loss_iter = np.zeros(n_iters, dtype=float)  # Temporary numpy array to save loss for each epoch
valid_loss_iter = np.zeros(n_iters, dtype=float)
train_accuracy_iter = np.zeros(n_iters, dtype=float)  # Temporary numpy array to save accuracy for each epoch
valid_accuracy_iter = np.zeros(n_iters, dtype=float)


def train(model, optimizer, train_loader, useGPU=True):
  model.train()
  total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

  # for each mini-batch
  n_known , n_word= 0.0, 0.0
  for batch_idx, sample_batched in enumerate(train_loader):
    label_tensor, text_tensor, update_bounds = sample_batched

    if useGPU:
      label_tensor = label_tensor.cuda()
      text_tensor = text_tensor.cuda()
      update_bounds = update_bounds.cuda()

    n_word += torch.numel(text_tensor)
    n_known += torch.count_nonzero(text_tensor).item()

    pred = model(text_tensor, update_bounds)
    optimizer.zero_grad()
    # print(pred.shape) # batch_size * n_classes

    loss = F.cross_entropy(pred, label_tensor)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() # accumulate loss
    total_cnt += label_tensor.size(0) # accumulate the number of data

    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct


  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  loss = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  known_rate = n_known * 1.0 / n_word

  # print(f"{known_rate} of words are known")
  return loss, accuracy

def validate(model, valid_loader, useGPU=True):
  model.eval()
  total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

  n_known , n_word= 0.0, 0.0
  for batch_idx, sample_batched in enumerate(valid_loader):
    with torch.no_grad():
      label_tensor, text_tensor, update_bounds = sample_batched
      if useGPU:
        label_tensor = label_tensor.cuda()
        text_tensor = text_tensor.cuda()
        update_bounds = update_bounds.cuda()

    n_word += torch.numel(text_tensor)
    n_known += torch.count_nonzero(text_tensor).item()

    pred = model(text_tensor, update_bounds)

    loss = F.cross_entropy(pred, label_tensor)

    total_loss += loss.item() # accumulate loss
    total_cnt += label_tensor.size(0) # accumulate the number of data
    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct
  
  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  loss = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  known_rate = n_known * 1.0 / n_word
  # print(f"{known_rate} of words are known")
  return loss, accuracy


# TODO 1회 training 하는데 3분 넘게 걸림 (batch_size=128)
for iter in range(n_iters):
  start = time.time()
  
  # train
  train_loss, train_accuracy = train(model, optimizer, train_loader)
  train_loss_iter[iter] =  train_loss
  train_accuracy_iter[iter] = train_accuracy

  # validation
  valid_loss, valid_accuracy = validate(model, valid_loader)
  valid_loss_iter[iter] =  valid_loss
  valid_accuracy_iter[iter] = valid_accuracy

  elapsed = time.strftime('%M min. %S sec.', time.localtime(time.time() - start))
  print(f"[{iter+1}/{n_iters}] Train Loss : {train_loss_iter[iter]:.4f} Train Acc : {train_accuracy_iter[iter]:.2f} \
  Valid Loss : {valid_loss_iter[iter]:.4f} Valid Acc : {valid_accuracy_iter[iter]:.2f} \
  Elapsed Time: {elapsed}")

torch.save(model, 'imdb-rnn-classification.pt')

# TODO test 함수 만들기