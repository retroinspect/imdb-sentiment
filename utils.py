import torch
import time
import math
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext import data, datasets
from torchtext.data import get_tokenizer
import os
import csv

unk_token = '<unk>'
pad_token = '<pad>'

tokenizer = get_tokenizer("basic_english")

def yield_tokens(text):
  # TODO preprocess
  # consider <br> </br>, special characters as space
  return tokenizer(text)

def build_vocab(train):
  voca_set = set()
  for _, (_, text) in enumerate(train):
    # print(text)
    tokens = set(yield_tokens(text))
    voca_set = voca_set.union(tokens)
  voca_list = [unk_token, pad_token] + list(voca_set)
  vocabulary = {k: v for v, k in enumerate(voca_list)}
  torch.save(vocabulary, 'vocab.pth')

def pad(text, max_len):
  text += [pad_token] * (max_len - len(text))
  return text

def preprocess(text):
  return text.strip().casefold().split()

def labelToTensor(label):
  if (label == 'pos'):
    val = 1
  else:
    val = 0
  return torch.FloatTensor([val, 1-val])


def thresholding(prediction):
  confidence, pred_label = torch.max(prediction, 1)
  return pred_label

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def split_data(dataset, train=0.4, val=0.1, test=0.5):
  total = len(dataset)
  n_train_data = int(total * train)
  n_val_data = int(total * val)
  n_test_data = total - n_train_data - n_val_data
  splited = random_split(dataset, [n_train_data, n_val_data, n_test_data])
  return splited

# TODO 만약 엄청나게 많은 데이터를 로드해야해서 메모리가 부족한 경우에는 어떻게 dataset을 짜야할까?

class IMDBDataset(Dataset):
  def __init__(self, datapath="IMDB Dataset.csv", transform=None):
    self.datapath = datapath
    self.transform = transform
    self.data = []

    with open(datapath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
              self.data.append({'text': row[0], 'label': row[1]})
              line_count += 1

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sample = self.data[idx]
    
    if self.transform:
      sample = self.transform(sample)
    
    return sample

