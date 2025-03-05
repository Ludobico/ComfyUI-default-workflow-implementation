from module.model_architecture import Unet
import torch
from module.model_state import load_unet_tensors, extract_model_components
from utils import get_torch_device


model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


unet = Unet.sdxl()
device = get_torch_device()
unet_tensors, clip_tensors, vae_tensors = extract_model_components(model_path)

updated_unet = load_unet_tensors(unet, unet_tensors)
