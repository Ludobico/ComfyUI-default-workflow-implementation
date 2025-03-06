from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
from typing import Dict
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.getenv import GetEnv
from module.model_architecture import Unet
from module.model_state import  extract_model_components
from utils import get_torch_device

def convert_unet_from_origin_sdxl(unet_sd : Dict, sdxl_unet_sd : Dict):
    path = ""
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(sdxl_unet_sd, unet_sd, path)
    return converted_unet_checkpoint

if __name__ == "__main__":
    env = GetEnv()


    model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


    unet = Unet.sdxl()
    device = get_torch_device()
    ckpt_unet_tensors, clip_tensors, vae_tensors = extract_model_components(model_path)

    converted = convert_unet_from_origin_sdxl(unet.state_dict(), ckpt_unet_tensors)