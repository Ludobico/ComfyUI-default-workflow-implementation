import os
from module.model_architecture import UNet, VAE, TextEncoder
import torch
from module.model_state import  extract_model_components
from utils import get_torch_device, highlight_print
from config.getenv import GetEnv
from module.module_utils import load_tokenizer
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from diffusers import StableDiffusionXLPipeline

env = GetEnv()


model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


unet = UNet.sdxl()
vae = VAE.sdxl_fp16()
enc1 = TextEncoder.sdxl_enc1()
device = get_torch_device()
ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(model_path)

converted_unet = convert_unet_from_ckpt_sd(unet, ckpt_unet_tensors)
converted_vae = convert_vae_from_ckpt_sd(vae, vae_tensors)
converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)



# print(type(converted_unet))
# print(type(converted_vae))
# print(type(converted_enc1), type(converted_enc2))
prompt = "A futuristic cityscape at night with neon lights and flying cars"
negative_prompt = "blurry, low quality"

tokenizer1, tokenizer2 = load_tokenizer(model_type)

pipe = StableDiffusionXLPipeline(
    unet=converted_unet,
    vae=converted_vae,
    text_encoder=converted_enc1,
    text_encoder_2=converted_enc2,
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    scheduler=None
)


image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024
)