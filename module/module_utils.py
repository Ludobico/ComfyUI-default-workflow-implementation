from typing import Literal, Optional
from safetensors.torch import load_file
from utils import get_cpu_device, get_torch_device, highlight_print
import torch
import os
from config.getenv import GetEnv

env = GetEnv()

def load_safetensors_file(ckpt, device : Literal['auto', 'gpu', 'cpu'] = 'auto'):
    """
    It works a safetensors file only.
    """
    if device == 'auto' or device == 'gpu':
        device = get_torch_device()
    elif device == 'cpu':
        device = get_cpu_device()
    else:
        device = get_torch_device()
    
    sd = load_file(ckpt, device=device)
    return sd

def get_model_keys(ckpt, save_name : Optional[str] = None, device : Literal['auto', 'gpu', 'cpu'] = 'auto'):
    
    if save_name is None:
        prefix_name = os.path.basename(ckpt).split('.')[0]
        save_name = f"{prefix_name}_keys.txt"
    
    save_path = os.path.join(env.get_output_dir(), save_name)
    sd = load_safetensors_file(ckpt, device=device)

    keys = '\n'.join(sorted(sd.keys()))

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(keys)
    
    highlight_print(f"Model keys are saved at : {save_path}", 'green')
    return True