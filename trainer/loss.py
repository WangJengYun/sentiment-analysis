import torch 
from torch import nn 
import torch.nn.functional as F

def CrossEntropyLoss(logits, labels, num_labels):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
    return loss