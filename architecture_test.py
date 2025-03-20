import os, pdb
from module.model_architecture import UNet, VAE, TextEncoder
import torch
from module.model_state import  extract_model_components
from utils import highlight_print
from config.getenv import GetEnv
from module.module_utils import load_tokenizer, limit_vram_usage, save_config_files
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from diffusers import StableDiffusionXLPipeline
from module.sampler.sampler_names import  scheduler_type
from module.debugging import pipe_from_diffusers
from module.encoder import PromptEncoder
from module.sampler.ksample_elements import retrieve_timesteps, prepare_latents
from module.torch_utils import get_torch_device, create_seed_generators

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

device = get_torch_device()
dtype = torch.float16
limit_vram_usage(device=device)

tokenizer1, tokenizer2 = load_tokenizer(model_type)

enc = PromptEncoder()
pos_prompt_embeds, pos_pooled_prompt_embeds = enc.sdxl_text_conditioning(prompt=prompt, clip=clip)
neg_prompt_embeds, neg_pooled_prompt_embeds = enc.sdxl_text_conditioning(prompt=negative_prompt, clip=clip)


scheduler = scheduler_type('euler_ancestral', 'normal')

timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=25, device=device)

num_channels_latents = unet.config.in_channels
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
# 1은 생성할 이미지 개수
# batch_size = pos_prompt_embeds.shape[0] * 1
# seed = 42
# generator = torch.Generator(device=device).manual_seed(seed)
batch_size = 1
generator = create_seed_generators(batch_size, seed=42, task='randomize')

if isinstance(prompt, str):
    latent_batch_size = 1 * batch_size
elif isinstance(prompt, list):
    latent_batch_size = len(prompt) * batch_size

# latent_batch_size (생성할 개수)가 2이상일수도 있으니 prompt_embeds 에서 따로 전처리필요, diffusers에는 encoding에서 수행하지만 comfy에서는 다른곳에서 수행
# 함수로 따로 만들어야할듯
bs_embed, seq_len, _ = pos_prompt_embeds.shape
pos_prompt_embeds = pos_prompt_embeds.repeat(1, batch_size, 1)
pos_prompt_embeds = pos_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)
pos_pooled_prompt_embeds = pos_pooled_prompt_embeds.repeat(1, batch_size).view(bs_embed * batch_size, -1)

bs_embed, seq_len, _ = neg_prompt_embeds.shape
neg_prompt_embeds = neg_prompt_embeds.repeat(1, batch_size, 1)
neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)
neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(1, batch_size).view(bs_embed * batch_size, -1)


latents = prepare_latents(latent_batch_size, num_channels_latents, 768, 1024, pos_prompt_embeds.dtype, torch.device(device), generator, vae_scale_factor)

pipe = StableDiffusionXLPipeline(
    unet=converted_unet,
    vae=converted_vae,
    text_encoder=clip[0],
    text_encoder_2=clip[1],
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    scheduler=scheduler
)
pipe.to(device=device, dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()


# 일반 문자열 프롬프트
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=25,
#     guidance_scale=7.5,
#     height=1024,
#     width=768,
#     generator=generator
# )

# implementation
image = pipe(
    prompt_embeds=pos_prompt_embeds,
    pooled_prompt_embeds=pos_pooled_prompt_embeds,
    negative_prompt_embeds=neg_prompt_embeds,
    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5,
    latents=latents,
    generator=generator
)

save_dir = env.get_output_dir()

if batch_size == 1:
    output = image.images[0]
    output.save(os.path.join(save_dir, 'output.png'))
elif batch_size > 1:
    for i, img in enumerate(image.images):
        save_path = os.path.join(save_dir, f"output_{i}.png")
        img.save(save_path)

print("DONE")