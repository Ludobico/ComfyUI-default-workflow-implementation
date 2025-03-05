import torch
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from pprint import pprint

if project_root not in sys.path:
    sys.path.append(project_root)
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from config.getenv import GetEnv
from module.loader.convert_diffusers_to_original_sdxl import convert_unet_state_dict, convert_vae_state_dict, convert_openclip_text_enc_state_dict

env = GetEnv()
ckpt_dir = env.get_checkpoint_model_dir()
test_ckpt_path = os.path.join(ckpt_dir, '[PONY]prefectPonyXL_v50.safetensors')

state_dict = load_file(test_ckpt_path)

