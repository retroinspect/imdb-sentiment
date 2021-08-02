"""
miscellaneous methods
"""

import torch
from torchtext.data import get_tokenizer
from tqdm import tqdm
import numpy as np
import re

pad_token = '<pad>'

def preprocess_text(word):
  word = re.sub(r"[^\w\s]", '', word)
  word = re.sub(r"\s+", ' ', word) # replace multiple spaces into single space
  word = re.sub(r"\d", '', word)  # remove digit
  return word

def tokenize_text(text):
  tokenized = get_tokenizer("basic_english")(text)
  return tokenized

def build_vocab(train, min_freq=5):
  words = []

  for _, sample in enumerate(tqdm(train, desc=f"tokenizing train set")):
    text = sample['text']
    preprocessed = preprocess_text(text)
    words += tokenize_text(preprocessed)

  print("counting words frequency")
  words = np.array(words, dtype=np.unicode_)
  unique, counts = np.unique(words, return_counts=True)
  vocabulary = dict({pad_token: 0})
  id = 1
  print(f"sieving words appeared at least {min_freq} times")
  for freq, word in zip(counts, unique): 
    if (freq < min_freq):
      continue
    vocabulary[word] = id
    id += 1

  return vocabulary

def pad(text, max_len):
  pads = [pad_token] * (max_len - len(text))
  # text = pads + text # Built-in GRU, LSTM
  text = text + pads # My RNN
  return text

def thresholding(prediction):
  confidence, pred_label = torch.max(prediction, dim=1)
  return pred_label