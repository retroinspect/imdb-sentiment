import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import thresholding

def validate(model, valid_loader, useGPU=True):
  model.eval()
  total_loss, correct_cnt = 0.0, 0.0
  n_known, n_word, n_padding = 0.0, 0.0, 0.0

  for batch_idx, sample_batched in enumerate(tqdm(valid_loader, desc="validating...")):
    with torch.no_grad():
      label_tensor, text_tensor, update_bounds = sample_batched

      if useGPU:
        label_tensor = label_tensor.cuda()
        text_tensor = text_tensor.cuda()
        update_bounds = update_bounds.cuda()

    n_word += torch.numel(text_tensor)
    n_known += torch.count_nonzero(text_tensor).item()
    n_padding += (text_tensor == 1).sum().item()

    pred = model(text_tensor, update_bounds)
    loss = F.cross_entropy(pred, label_tensor)
    # loss = F.cross_entropy(logit, label_tensor, reduction='sum')

    total_loss += loss.item() # accumulate loss
    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct
  
  total_cnt = len(valid_loader.dataset)
  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  loss = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  known_rate = (n_known - n_padding) * 1.0 / (n_word - n_padding)
  return loss, accuracy, known_rate