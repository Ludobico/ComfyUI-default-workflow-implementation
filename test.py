from config.getenv import GetEnv
from utils import highlight_print
from module.model_state import get_model_keys, extract_model_components
from module.module_utils import auto_model_detection
import os
from module.model_architecture import Unet
env = GetEnv()

model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"

# check tensors from safetensors
# unet_tensors, clip_tensors, vae_tensors = extract_model_components(model_path)

unet = Unet.sdxl()
print(unet.state_dict().keys())

