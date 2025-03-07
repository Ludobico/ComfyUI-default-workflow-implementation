import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
from typing import Dict, Literal
from config.getenv import GetEnv
from module.model_architecture import UNet, VAE, TextEncoder
from module.model_state import  extract_model_components
from utils import get_torch_device, highlight_print
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from config.getenv import GetEnv
from diffusers import UNet2DConditionModel, AutoencoderKL

env = GetEnv()

def convert_unet_from_ckpt_sd(unet : UNet2DConditionModel, ckpt_unet_sd : Dict):
    path = ""
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(ckpt_unet_sd, unet.config, path)
    unet.load_state_dict(converted_unet_checkpoint)
    return unet

def convert_vae_from_ckpt_sd(vae : AutoencoderKL, ckpt_vae_sd : Dict ):
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(ckpt_vae_sd, vae.config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae

def convert_clip_from_ckpt_sd(clip_model : CLIPTextModel, ckpt_clip_sd : Dict, model_type : Literal['sd15', 'sdxl']):
    converted_clip1_checkpoint = convert_ldm_clip_checkpoint(ckpt_clip_sd, text_encoder=clip_model)

    config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    if model_type == 'sd15':
        prefix = "cond_stage_model.model."
    elif model_type == 'sdxl':
        prefix = "conditioner.embedders.1.model."
    cache_dir = env.get_clip_model_dir()
    config_kwargs = {"projection_dim" : 1280, "cache_dir" : cache_dir}


    converted_clip2_checkpoint = convert_open_clip_checkpoint(ckpt_clip_sd, config_name, prefix=prefix, has_projection=True, **config_kwargs)
    return converted_clip1_checkpoint, converted_clip2_checkpoint

def get_clip_to_update_state(clip_model , clip_model2, ckpt_clip_sd : Dict, model_type : Literal['sd15', 'sdxl']):
    pass

# def convert_clip2_from_ckpt_sd(clip_config , ckpt_clip_sd : Dict, model_type : Literal['sd15', 'sdxl']):
#     if model_type == 'sd15':
#         prefix = "cond_stage_model.model."
#     elif model_type == 'sdxl':
#         prefix = "conditioner.embedders.1.model."
#     converted_clip_checkpoint = convert_open_clip_checkpoint(ckpt_clip_sd, clip_config, prefix=prefix)
#     return converted_clip_checkpoint

if __name__ == "__main__":
    env = GetEnv()


    model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


    unet = UNet.sdxl()
    vae = VAE.sdxl_fp16()
    enc1 = TextEncoder.sdxl_enc1()
    device = get_torch_device()
    ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(model_path)

    converted_unet = convert_unet_from_ckpt_sd(unet.config, ckpt_unet_tensors)
    converted_vae = convert_vae_from_ckpt_sd(vae.config, vae_tensors)
    converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)