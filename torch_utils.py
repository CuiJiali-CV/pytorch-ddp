import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import torch

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_ckpt(state_dict, save_path):
    torch.save(state_dict, save_path)

def requires_grad(model, flag=True):
    params = model.parameters()
    for p in params:
        p.requires_grad = flag