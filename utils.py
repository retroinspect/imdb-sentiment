"""
miscellaneous methods
"""

import torch
from torchtext.data import get_tokenizer
from tqdm import tqdm
import numpy as np
import re

unk_token = '<unk>'
pad_token = '<pad>'

class MyTokenizer(object):
  def _preprocess_string(self, s):
    s = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', s) # remove special characters
    s = re.sub(r"\s+", ' ', s) # replace multiple spaces into single space
    s = re.sub(r"\d", '', s) # remove digit
    return s

  def __init__(self, final_tokenizer="basic_english"):
    self.final_tokenizer = get_tokenizer(final_tokenizer)

  def __call__(self, string):
    preprocessed = self._preprocess_string(string)
    return self.final_tokenizer(preprocessed)


def build_vocab(train, min_freq=5):
  words = []
  tokenizer = MyTokenizer()

  for _, sample in enumerate(tqdm(train, desc=f"tokenizing train set")):
    text = sample['text']
    words += tokenizer(text)

  print("counting words frequency")
  words = np.array(words, dtype=np.unicode_)
  unique, counts = np.unique(words, return_counts=True)
  vocabulary = dict({unk_token: 0, pad_token: 1})
  id = 2
  print(f"sieving words appeared at least {min_freq} times")
  for freq, word in zip(counts, unique): 
    if (freq < min_freq):
      continue
    vocabulary[word] = id
    id += 1

  torch.save(vocabulary, 'vocab.pth')
  return vocabulary

def pad(text, max_len):
  text += [pad_token] * (max_len - len(text))
  return text

def thresholding(prediction):
  confidence, pred_label = torch.max(prediction, dim=1)
  return pred_label
