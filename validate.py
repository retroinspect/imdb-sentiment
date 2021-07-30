import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import thresholding

def validate(model, valid_loader, useGPU=True, desc="validating..."):
  model.eval()
  total_loss, correct_cnt = 0.0, 0.0

  for batch_idx, sample_batched in enumerate(tqdm(valid_loader, desc=desc)):
    with torch.no_grad():
      label_tensor, text_tensor, update_bounds = sample_batched

      if useGPU:
        label_tensor = label_tensor.cuda()
        text_tensor = text_tensor.cuda()
        update_bounds = update_bounds.cuda()

    pred = model(text_tensor, update_bounds)
    loss = F.cross_entropy(pred, label_tensor)
    # loss = F.cross_entropy(logit, label_tensor, reduction='sum')

    total_loss += loss.item() # accumulate loss
    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct
  
  total_cnt = len(valid_loader.dataset)
  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  loss = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  return loss, accuracy