import os, pdb
from module.model_architecture import UNet, VAE, TextEncoder
import torch
from module.model_state import  extract_model_components
from utils import get_torch_device, highlight_print
from config.getenv import GetEnv
from module.module_utils import load_tokenizer, limit_vram_usage, save_config_files
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from diffusers import StableDiffusionXLPipeline
from module.sampler.sampler_names import euler_ancestral, scheduler_type
from module.debugging import pipe_from_diffusers
from module.encoder import sdxl_text_conditioning

env = GetEnv()
torch.cuda.empty_cache()

model_path = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\[SDXL]sd_xl_base_1.0.safetensors"

model_path2 = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"


unet = UNet.sdxl()
vae = VAE.sdxl()
enc1 = TextEncoder.sdxl_enc1()
device = get_torch_device()
limit_vram_usage(device=device)
ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(model_path)

converted_unet = convert_unet_from_ckpt_sd(unet, ckpt_unet_tensors)
converted_vae = convert_vae_from_ckpt_sd(vae, vae_tensors)
converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)
clip = (converted_enc1, converted_enc2)

prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
negative_prompt = "text, watermark"

emb_prompt = sdxl_text_conditioning(prompt, clip, 'cpu')
highlight_print(emb_prompt, 'green')
pdb.set_trace()
seed = 42
device = get_torch_device()
limit_vram_usage(device=device)
generator = torch.Generator(device=device).manual_seed(seed)

tokenizer1, tokenizer2 = load_tokenizer(model_type)

schedular = scheduler_type(euler_ancestral, 'normal')

pipe = StableDiffusionXLPipeline(
    unet=converted_unet,
    vae=converted_vae,
    text_encoder=clip[0],
    text_encoder_2=clip[1],
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    scheduler=schedular
)
pipe.to(device=device, dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

# 일반 문자열 프롬프트
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=1024,
    width=768,
    generator=generator
)

output = image.images[0]
save_path = env.get_output_dir()

output.save(os.path.join(save_path, 'output.png'))
print("DONE")