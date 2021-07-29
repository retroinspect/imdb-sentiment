import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import thresholding

def train(model, optimizer, train_loader, useGPU=True, clip=5):
  model.train()
  total_loss, correct_cnt = 0.0, 0.0

  for batch_idx, sample_batched in enumerate(tqdm(train_loader, desc="training...")):
    label_tensor, text_tensor, update_bounds = sample_batched

    if useGPU:
      label_tensor = label_tensor.cuda()
      text_tensor = text_tensor.cuda()
      update_bounds = update_bounds.cuda()

    optimizer.zero_grad()
    pred = model(text_tensor, update_bounds) # shape: batch_size * n_classes

    loss = F.cross_entropy(pred, label_tensor)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    total_loss += loss.item()
    correct_cnt += (label_tensor == thresholding(pred)).sum().item()

  total_cnt = len(train_loader.dataset)
  accuracy = correct_cnt * 1.0 / total_cnt 
  loss = total_loss / total_cnt

  return loss, accuracy


