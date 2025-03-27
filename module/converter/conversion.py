import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from module.converter.sd_to_diffuser import convert_unet, convert_vae, convert_text_encoder, convert_text_encoder_2
from typing import Dict, Literal
from config.getenv import GetEnv
from module.model_architecture import TextEncoder
from transformers import CLIPTextModel
from config.getenv import GetEnv
from diffusers import UNet2DConditionModel, AutoencoderKL

env = GetEnv()

def convert_unet_from_ckpt_sd(unet : UNet2DConditionModel, ckpt_unet_sd : Dict):
    path = ""
    # converted_unet_checkpoint = convert_ldm_unet_checkpoint(ckpt_unet_sd, unet.config, path)
    converted_unet_checkpoint = convert_unet(ckpt_unet_sd, unet.config, path)

    unet.load_state_dict(converted_unet_checkpoint, strict=False)
    return unet

def convert_vae_from_ckpt_sd(vae : AutoencoderKL, ckpt_vae_sd : Dict ):
    # converted_vae_checkpoint = convert_ldm_vae_checkpoint(ckpt_vae_sd, vae.config)
    converted_vae_checkpoint = convert_vae(ckpt_vae_sd, vae.config)

    vae.load_state_dict(converted_vae_checkpoint, strict=False)
    return vae

def convert_clip_from_ckpt_sd(clip_model : CLIPTextModel, ckpt_clip_sd : Dict, model_type : Literal['sd15', 'sdxl']):
    # converted_clip1_checkpoint = convert_ldm_clip_checkpoint(ckpt_clip_sd, text_encoder=clip_model)
    converted_clip1_checkpoint = convert_text_encoder(ckpt_clip_sd, text_encoder=clip_model)

    if model_type == 'sd15':
        prefix = "cond_stage_model.model."
        return converted_clip1_checkpoint
    
    elif model_type == 'sdxl':
        prefix = "conditioner.embedders.1.model."
        # cache_dir = env.get_clip_model_dir()
        # config_kwargs = {"projection_dim" : 1280, "cache_dir" : cache_dir}
        config = TextEncoder.sdxl_enc2_config()


        # converted_clip2_checkpoint = convert_open_clip_checkpoint(ckpt_clip_sd, config_name, prefix=prefix, has_projection=True, **config_kwargs)
        converted_clip2_checkpoint = convert_text_encoder_2(ckpt_clip_sd, config, prefix=prefix, has_projection=True, local_files_only = True)
        return (converted_clip1_checkpoint, converted_clip2_checkpoint)

