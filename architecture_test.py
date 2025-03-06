import os
from module.model_architecture import Unet
import torch
from module.model_state import  extract_model_components
from utils import get_torch_device
from config.getenv import GetEnv
from module.module_utils import compare_unet_models
from module.converter.conversion import convert_unet_from_origin_sdxl

env = GetEnv()


model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


unet = Unet.sdxl()
device = get_torch_device()
ckpt_unet_tensors, clip_tensors, vae_tensors = extract_model_components(model_path)

converted = convert_unet_from_origin_sdxl(unet.state_dict(), ckpt_unet_tensors)