import os
from module.model_architecture import UNet, VAE, TextEncoder
import torch
from module.model_state import  extract_model_components
from utils import get_torch_device, highlight_print
from config.getenv import GetEnv
from module.module_utils import load_tokenizer, limit_vram_usage
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from diffusers import StableDiffusionXLPipeline
from module.sampler.sampler_names import euler_ancestral, schedular_type

env = GetEnv()


model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


unet = UNet.sdxl()
vae = VAE.sdxl_fp16()
enc1 = TextEncoder.sdxl_enc1()
device = get_torch_device()
limit_vram_usage(device=device)
ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(model_path)

converted_unet = convert_unet_from_ckpt_sd(unet, ckpt_unet_tensors)
converted_vae = convert_vae_from_ckpt_sd(vae, vae_tensors)
converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)



# print(type(converted_unet))
# print(type(converted_vae))
# print(type(converted_enc1), type(converted_enc2))
prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
negative_prompt = "text, watermark"

tokenizer1, tokenizer2 = load_tokenizer(model_type)

schedular = schedular_type(euler_ancestral, 'normal')

pipe = StableDiffusionXLPipeline(
    unet=converted_unet,
    vae=converted_vae,
    text_encoder=converted_enc1,
    text_encoder_2=converted_enc2,
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    scheduler=schedular
)

pipe.to(device)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024
)