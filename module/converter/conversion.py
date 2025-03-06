from module.converter import convert_diffusers_to_original_sdxl as csdxl
from typing import Dict
from diffusers import StableDiffusionXLPipeline
def convert_sdxl_to_diffusers(tensors_dict :Dict):
    unet_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in tensors_dict.items() if k.startswith("model.diffusion_model.")}
    unet_state_dict = csdxl.convert_unet_state_dict(unet_state_dict)

    # vae_state_dict = {k.replace("first_stage_model.", ""): v for k, v in tensors_dict.items() if k.startswith("first_stage_model")}
    # vae_state_dict = csdxl.revese_convert_vae_sd(vae_state_dict)

    # text_enc_dict = {k.replace("conditioner.embedders.0.transformer.", ""): v for k, v in tensors_dict.items() if k.startswith("conditioner.embedders.0.transformer")}
    # text_enc_dict = csdxl.reverse_convert_clip_sd(text_enc_dict)

    # text_enc_2_dict = {k.replace("conditioner.embedders.1.model.", ""): v for k, v in tensors_dict.items() if k.startswith("conditioner.embedders.1.model")}
    # text_enc_2_dict = csdxl.reverse_convert_clip_sd(text_enc_2_dict)

    return {
        "unet": unet_state_dict,
        # "vae": vae_state_dict,
        # "text_encoder": text_enc_dict,
        # "text_encoder_2": text_enc_2_dict
    }

def convert_sdxl_to_diffusers():
    pass