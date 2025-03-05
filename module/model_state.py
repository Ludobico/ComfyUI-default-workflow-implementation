import os
from typing import Literal, Optional
from safetensors.torch import load_file
from utils import get_cpu_device, get_torch_device, highlight_print
import torch
from config.getenv import GetEnv

env = GetEnv()

def get_model_keys(ckpt : os.PathLike, return_type : Literal['str', 'list'] = 'str',save_as_file : bool = True ,save_name : Optional[str] = None, device : Literal['auto', 'gpu', 'cpu'] = 'cpu'):
    from module.module_utils import load_safetensors_file
    sd = load_safetensors_file(ckpt, device=device)

    if return_type == 'str':
        keys = '\n'.join(sorted(sd.keys()))
    elif return_type == 'list':
        keys = sorted(sd.keys())

    if save_as_file:
        if save_name is None:
            prefix_name = os.path.basename(ckpt).split('.')[0]
            save_name = f"{prefix_name}_keys.txt"

        save_path = os.path.join(env.get_output_dir(), save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(keys)
        
        highlight_print(f"Model keys are saved at : {save_path}", 'green')
    return keys

def is_unet_tensor(key, model_type : Literal['sd15', 'sdxl']):
    if model_type == 'sd15':
        return key.startswith("model.diffusion_model.")
    elif model_type == 'sdxl':
        return key.startswith("model.diffusion_model.")
    return False

def is_clip_tensor(key, model_type : Literal['sd15', 'sdxl']):
    if model_type == 'sd15':
        return key.startswith("cond_stage_model.transformer.")
    elif model_type == 'sdxl':
        return key.startswith("conditioner.embedders")
    return False

def is_vae_tensor(key, model_type : Literal['sd15', 'sdxl']):
    if model_type == 'sd15':
        return key.startswith("first_stage_model.")
    elif model_type == 'sdxl':
        return key.startswith("first_stage_model.")
    return False