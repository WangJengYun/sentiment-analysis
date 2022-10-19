import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Iterable, Optional, Tuple, Union

def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    return LambdaLR(optimizer, lambda _:1, last_epoch = last_epoch)

def get_constant_schedule_with_warmup(
    optimizer: Optimizer, 
    num_epoch: int,
    num_training_steps_per_epoch: int,
    num_warmup_steps: int = None,
    last_epoch: int = -1
    ):

    if num_warmup_steps is None:
        num_warmup_steps = int(num_epoch//10 * num_training_steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(current_step, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch = last_epoch) 

def get_consine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_epoch: int,
    num_training_steps_per_epoch: int,
    num_warmup_steps: int = None,
    num_training_steps: int = None,
    num_cycles: float = 0.5,
    last_epoch: int = -1
    ):

    if num_warmup_steps is None:
        num_warmup_steps = int(num_epoch//10 * num_training_steps_per_epoch)

    if num_training_steps is None:
        num_training_steps = int(num_epoch * num_training_steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(current_step, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress )))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
