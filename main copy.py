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
# dataset = IMDBDataset(datapath='imdb_binary_sent.csv')
# train_set, valid_set, test_set = split_data(dataset)
# print(f'train set size: {len(train_set)}')
# print(f'validation set size: {len(valid_set)}')
# print(f'test set size: {len(test_set)}')
import pandas as pd
from sklearn.model_selection import train_test_split

base_csv = './imdb_binary_sent.csv'
df = pd.read_csv(base_csv)
df.head()
X,y = df['review'].values, df['sentiment'].values
x_train,x_val,y_train,y_val = train_test_split(X,y,stratify=y)
print(f'length of train data is {len(x_train)}')
print(f'length of test data is {len(x_val)}')


""" 
build vocabulary & dataloader
"""
# try:
#   vocabulary = torch.load('vocab.pth')
# except:
#   vocabulary = build_vocab(train_set)

# vocab_size = len(vocabulary)
# print(f'vocab size: {vocab_size}')
# collate_fn = MyCollator(vocabulary)
# train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
import re
from nltk.corpus import stopwords 
from collections import Counter
import numpy as np # linear algebra
# from torch.utils.data import TensorDataset, DataLoader

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", '', s)
    return s

def build_vocab(x_train):
  word_list = []
  stop_words = set(stopwords.words('english')) 
  for sent in x_train:
      for word in sent.lower().split():
          word = preprocess_string(word)
          if word not in stop_words and word != '':
              word_list.append(word)

  corpus = Counter(word_list)
  corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:20000]
  vocab = {w:i+1 for i,w in enumerate(corpus_)}
  return vocab

def lookup_vocab(word, vocab):
  if word in vocab.keys():
    return vocab[word]
  return 0

def encode(x_train, y_train, x_val, y_val, vocab):
  encoded_x_train, encoded_x_val = [], []
  
  for sent in x_train:
    encoded_x_train.append([lookup_vocab(word, vocab) for word in sent])

  for sent in x_val:
    encoded_x_val.append([lookup_vocab(word, vocab) for word in sent])

  encoded_x_train, encoded_x_val = np.array(encoded_x_train), np.array(encoded_x_val)
  encoded_y_train = np.array([1 if label =='positive' else 0 for label in y_train])
  encoded_y_val = np.array([1 if label =='positive' else 0 for label in y_val])
  
  return encoded_x_train, encoded_y_train, encoded_x_val, encoded_y_val

def tokenize(x_train, x_val):
  final_list_train, final_list_test = [], []
  for sent in x_train:
    if (type(sent) == type('string')):
      final_list_train.append([preprocess_string(word) for word in sent.lower().split()])
    else:
      print(f"type(sent): {type(sent)}")
  for sent in x_val:
    final_list_test.append([preprocess_string(word) for word in sent.lower().split()])
  return final_list_train, final_list_test

try:
  vocab = torch.load('vocab.pth')
except:
  vocab = build_vocab(x_train)
  torch.save(vocab, 'vocab.pth')

vocab_size = len(vocab) + 1
print(f'Length of vocabulary is {len(vocab)}')

try:
  x_train, y_train, x_val, y_val = torch.load('data.pth')
except:
  x_train, x_val = tokenize(x_train, x_val)
  x_train, y_train, x_val, y_val = encode(x_train, y_train, x_val, y_val, vocab)
  torch.save((x_train, y_train, x_val, y_val, vocab), 'data.pth')


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# TODO batch collate로 batch마다만 패딩이 있도록 해보기

class TrainDataset(Dataset):
  def __init__(self):
    self.data = list(zip(x_train, y_train))

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    return self.data[idx]

class TestDataset(Dataset):
  def __init__(self):
    self.data = list(zip(x_val, y_val))

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    return self.data[idx]

# x_train_pad = padding_(x_train,500)
# x_val_pad = padding_(x_val,500)

# train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
# valid_data = TensorDataset(torch.from_numpy(x_val_pad), torch.from_numpy(y_val))

def collate_fn(batch):
  texts, labels = [], []
  for sample in batch:
    text, label = sample
    texts.append(text)
    labels.append(label)

  max_len = max([len(text) for text in texts])

  padded = padding_(texts, max_len)
  texts = torch.from_numpy(padded) # texts.shape : max_len * batch_size
  labels = torch.LongTensor(labels)
  return (texts, labels)

train_data = TrainDataset()
valid_data = TestDataset()

valid_loader = DataLoader(valid_data, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
train_loader = DataLoader(train_data, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)

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