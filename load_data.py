
"""
methods & classes on dataset, batching
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from utils import pad, preprocess_text, tokenize_text

def split_data(dataset, train=0.4, val=0.1, test=0.5):
  total = len(dataset)
  n_train_data = int(total * train)
  n_val_data = int(total * val)
  n_test_data = total - n_train_data - n_val_data
  splited = random_split(dataset, [n_train_data, n_val_data, n_test_data])
  return splited

class IMDBDataset(Dataset):
  def __init__(self, datapath="imdb_binary_sent.csv", transform=None):
    self.datapath = datapath
    self.transform = transform
    df = pd.read_csv(datapath)  
    df.head()

    review, sentiment = df['review'].values, df['sentiment'].values
    iter = zip(review, sentiment)
    self.data = [{'label': label, 'text': text} for _, (text, label) in enumerate(iter)]

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sample = self.data[idx]
    
    if self.transform:
      sample = self.transform(sample)
    
    return sample


class MyCollator(object):
  def _tokenToLong(self, token):
    if not token in self.vocabulary:
      item = 0
    else:
      item = self.vocabulary[token]
    return item

  def _textToLong(self, text):
    return [self._tokenToLong(token) for token in text]

  def __init__(self, vocabulary):
    self.vocabulary = vocabulary
    
  def __call__(self, batch): # batch example: {label: 'positive', text: 'good movie'}
    labels, texts = [], []
    for sample in batch:
      label, text = sample['label'], sample['text']
      preprocessed = preprocess_text(text)
      tokenized = tokenize_text(preprocessed)
      texts.append(tokenized)
      labels.append(label)
    
    text_lengths = torch.LongTensor([len(text) for text in texts])
    max_len = max(text_lengths)

    texts = torch.LongTensor([self._textToLong(pad(text, max_len)) for text in texts]) # texts.shape : max_len * batch_size
    labels = torch.LongTensor([label == 'positive' for label in labels]) # labels.shape : batch_size

    text_lengths, sorted_idx = text_lengths.sort(0, descending=True)
    texts = texts[sorted_idx]
    texts = torch.transpose(texts, 0, 1)
    labels = labels[sorted_idx]

    update_bounds = torch.LongTensor([0] * max_len) # number of words being processed at once
    for text_len in text_lengths:
      tmp = torch.LongTensor([1] * text_len + [0] * (max_len - text_len))
      update_bounds += tmp;

    return (labels, texts, update_bounds)