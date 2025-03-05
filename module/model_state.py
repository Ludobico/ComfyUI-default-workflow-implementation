import os
from typing import Literal, Optional, Dict
from safetensors.torch import load_file
from utils import get_cpu_device, get_torch_device, highlight_print
import torch
from config.getenv import GetEnv
from module.module_utils import load_checkpoint_file, auto_model_detection
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

env = GetEnv()

def get_model_keys(ckpt : os.PathLike, return_type : Literal['str', 'list'] = 'str',save_as_file : bool = False ,save_name : Optional[str] = None, device : Literal['auto', 'gpu', 'cpu'] = 'cpu'):
    sd = load_checkpoint_file(ckpt, device=device)

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

def extract_model_components(ckpt):
    unet_tensors = {}
    clip_tensors = {}
    vae_tensors = {}

    model_type = auto_model_detection(ckpt)
    tensors = load_checkpoint_file(ckpt, device='cpu')

    for key, tensor in tensors.items():
        if is_unet_tensor(key, model_type):
            unet_tensors[key] = tensor.cpu()
        elif is_clip_tensor(key, model_type):
            clip_tensors[key] = tensor.cpu()
        elif is_vae_tensor(key, model_type):
            vae_tensors[key] = tensor.cpu()
    
    return unet_tensors, clip_tensors, vae_tensors


def load_unet_tensors(unet_model : UNet2DConditionModel, unet_tensors : Dict):
    model_state_dict = unet_model.state_dict()

    updated_state_dict = {}
    for key, tensor in unet_tensors.items():
        if key in model_state_dict:
            if tensor.shape == model_state_dict[key].shape:
                updated_state_dict[key] = tensor
            else:
                print(f"Shape mismatch for {key} : expected {model_state_dict[key].shape}")
        else:
            print(f"key {key} not found in model state_dict")
    
    model_state_dict.update(updated_state_dict)
    unet_model.load_state_dict(model_state_dict)
    return unet_model


