import torch 
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class SequenceClassifierOutput(object):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None