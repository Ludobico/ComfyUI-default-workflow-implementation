import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
from typing import Dict, Union
from config.getenv import GetEnv
from module.model_architecture import UNet, VAE, TextEncoder
from module.model_state import  extract_model_components
from utils import get_torch_device, highlight_print
from transformers import CLIPTextModel, CLIPTextModelWithProjection

def convert_unet_from_ckpt_sd(unet_config : Dict, ckpt_unet_sd : Dict):
    path = ""
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(ckpt_unet_sd, unet_config, path)
    return converted_unet_checkpoint

def convert_vae_from_ckpt_sd(vae_config : Dict, ckpt_vae_sd : Dict ):
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(ckpt_vae_sd, vae_config)
    return converted_vae_checkpoint

def convert_clip_from_ckpt_sd(clip_model : CLIPTextModel, ckpt_clip_sd : Dict):
    converted_clip_checkpoint = convert_ldm_clip_checkpoint(ckpt_clip_sd, text_encoder=clip_model)
    return converted_clip_checkpoint

def convert_clip2_from_ckpt_sd(clip_model : CLIPTextModelWithProjection, ckpt_clip_sd : Dict):
    pass

if __name__ == "__main__":
    env = GetEnv()


    model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


    unet = UNet.sdxl()
    vae = VAE.sdxl_fp16()
    enc1 = TextEncoder.sdxl_enc1()
    enc2 = TextEncoder.sdxl_enc2()
    device = get_torch_device()
    ckpt_unet_tensors, clip_tensors, vae_tensors = extract_model_components(model_path)

    # converted_unet = convert_unet_from_ckpt_sd(unet.config, ckpt_unet_tensors)
    # converted_vae = convert_vae_from_ckpt_sd(vae.config, vae_tensors)
    # converted_enc1 = convert_clip_from_ckpt_sd(enc1, clip_tensors)
    converted_enc2 = convert_clip_from_ckpt_sd(enc2, clip_tensors)