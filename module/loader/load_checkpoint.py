import torch
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from config.getenv import GetEnv
from utils import get_torch_device
from typing import Literal

def is_unet_tensor(key, model_type : Literal['sd15', 'sdxl', 'flux']):
    if model_type == 'sd15':
        return key.startswith("model.diffusion_model.")
    elif model_type == 'sdxl':
        return key.startswith("model.diffusion_model.")
    elif model_type == 'flux':
        return any(key.startswith(prefix) for prefix in [
            "unet.", "diffusion_model.", "model.diffusion_model.",
            "double_blocks.", "single_blocks.", "final_layer.",
            "guidance_in.", "img_in."
        ])
    return False

def get_unet_tensors(ckpt : os.PathLike, model_type : Literal['sd15', 'sdxl', 'flux']):
    device = get_torch_device()
    unet_tensor = {}

    with safe_open(ckpt, framework='pt', device=device) as f:
        for key in f.keys():
            if is_unet_tensor(key, model_type):
                tensor = f.get_tensor(key)
                if model_type == 'sd15':
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_tensor[new_key] = tensor.cpu()
                else:
                    unet_tensor[key] = tensor.cpu()
    return unet_tensor



env = GetEnv()
ckpt_dir = env.get_checkpoint_model_dir()
test_ckpt_path = os.path.join(ckpt_dir, '[PONY]prefectPonyXL_v50.safetensors')
unet_tensors = get_unet_tensors(test_ckpt_path, 'sdxl')

project_dir = env.get_project_dir()
output_file = "unet_keys.txt"

with open(os.path.join(project_dir, output_file), 'w', encoding='utf-8') as f:
    f.write("\n".join(unet_tensors.keys()))