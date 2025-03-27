import os
from typing import Literal, Optional, Dict, Tuple
from utils import highlight_print
from config.getenv import GetEnv
from module.module_utils import load_checkpoint_file, auto_model_detection

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

def split_sdxl_clip_tensors(clip_tensors : Dict):
    """
    clip1 : CLIPTextModel clip-vit-large-patch14 variant.
    clip2 : CLIPTextModelWithProjection laion/CLIP-ViT-bigG-14-laion2B-39B-b160k 
    """
    

def extract_model_components(ckpt) -> Tuple[Dict, Dict, Dict]:
    """
    Extracts UNet, CLIP, VAE weights and model type(sd15 or sdxl) from a checkpoint as a dictionary\n
    """
    unet_tensors = {}
    clip_tensors = {}
    vae_tensors = {}

    model_type = auto_model_detection(ckpt)
    tensors = load_checkpoint_file(ckpt, device='cpu')

    for key, tensor in tensors.items():
        if is_unet_tensor(key, model_type):
            unet_tensors[key] = tensor
        elif is_clip_tensor(key, model_type):
            clip_tensors[key] = tensor
        elif is_vae_tensor(key, model_type):
            vae_tensors[key] = tensor
    
    return unet_tensors, clip_tensors, vae_tensors, model_type