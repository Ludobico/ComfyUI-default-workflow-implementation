import os
from typing import Literal, Optional, Dict
from safetensors.torch import load_file
from utils import get_cpu_device, get_torch_device, highlight_print
from module.model_state import get_model_keys, is_clip_tensor
import torch
from config.getenv import GetEnv

env = GetEnv()

def load_safetensors_file(ckpt, device : Literal['auto', 'gpu', 'cpu'] = 'auto'):
    """
    Load a model file (either .safetensors or .ckpt).
    """
    if device == 'auto' or device == 'gpu':
        device = get_torch_device()
    elif device == 'cpu':
        device = get_cpu_device()
    else:
        device = get_torch_device()
    
    ext = os.path.splitext(ckpt)[-1].lower()

    if ext == '.safetensors':
        sd = load_file(ckpt, device=device)
    elif ext == '.ckpt':
        model = torch.load(ckpt, map_location=device)
        if "state_dict" in model:
            sd = model['state_dict']
        else:
            sd = model
    return sd

def auto_model_detection(ckpt) -> str:
    """
    Is it stable diffusion? or SDXL?
    """
    sd = get_model_keys(ckpt, return_type='list', save_as_file=False)

    prefix_sd = [[part for part in item.split('.')[:1]] for item in sd]
    
    def flatten(lst):
        result = []
        for item in lst:
            result.extend(item)
        return result
    
    # Flatten a nested lits into a single list
    preprocess_sd1 = flatten(prefix_sd)
    # Remove duplicates from list
    preprocess_sd2 = list(set(preprocess_sd1))
    
    if preprocess_sd2 in 'cond_stage_model':
        return 'sd15'
    
    if preprocess_sd2 in 'conditioner':
        return 'sdxl'
    
    return False




