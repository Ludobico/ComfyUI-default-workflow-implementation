import os, pdb
from module.model_architecture import UNet, VAE, TextEncoder
import torch
from module.model_state import  extract_model_components
from utils import  highlight_print
from config.getenv import GetEnv
from module.module_utils import load_tokenizer, save_config_files
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from diffusers import StableDiffusionPipeline
from module.sampler.sampler_names import scheduler_type
from module.debugging import pipe_from_diffusers
from module.encoder import PromptEncoder, sd_clip_postprocess
from module.sampler.ksample_elements import retrieve_timesteps, prepare_latents
from module.torch_utils import get_torch_device, create_seed_generators, limit_vram_usage
from diffusers.image_processor import VaeImageProcessor
import inspect

env = GetEnv()
torch.cuda.empty_cache()

model_path = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\dreamshaper_8.safetensors"


unet = UNet.sd15()
vae = VAE.sd15()
enc1 = TextEncoder.sd15_enc()
device = get_torch_device()
limit_vram_usage(device=device)
ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(model_path)

converted_unet = convert_unet_from_ckpt_sd(unet, ckpt_unet_tensors)
converted_vae = convert_vae_from_ckpt_sd(vae, vae_tensors)
converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)
clip = converted_enc1

prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
negative_prompt = "text, watermark"

device = get_torch_device()
dtype = torch.float16
limit_vram_usage(device=device)

tokenizer1 = load_tokenizer(model_type)

enc = PromptEncoder()
pos_prompt_embeds = enc.sd15_text_conditioning(prompt=prompt, clip=clip)
neg_prompt_embeds = enc.sd15_text_conditioning(prompt=negative_prompt, clip=clip)


scheduler = scheduler_type('dpmpp_2m', 'karras')
original_timesteps = scheduler.timesteps
        # 8.1 Apply denoising_end
        # if (
        #     self.denoising_end is not None
        #     and isinstance(self.denoising_end, float)
        #     and self.denoising_end > 0
        #     and self.denoising_end < 1
        # ):
        #     discrete_timestep_cutoff = int(
        #         round(
        #             self.scheduler.config.num_train_timesteps
        #             - (self.denoising_end * self.scheduler.config.num_train_timesteps)
        #         )
        #     )
        #     num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        #     timesteps = timesteps[:num_inference_steps]

timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=25, device=device)

num_channels_latents = unet.config.in_channels
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
# 1은 생성할 이미지 개수
batch_size = 1
generator = create_seed_generators(batch_size, task='fixed', seed=42)

pos_prompt_embeds = sd_clip_postprocess(pos_prompt_embeds, batch_size)
neg_prompt_embeds = sd_clip_postprocess(neg_prompt_embeds, batch_size)

latents = prepare_latents(batch_size, num_channels_latents, 512, 512, pos_prompt_embeds.dtype, torch.device(device), generator, vae_scale_factor)

pipe = StableDiffusionPipeline(
    unet=converted_unet,
    vae=converted_vae,
    text_encoder=clip,
    tokenizer=tokenizer1,
    scheduler=scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
)
pipe.to(device=device, dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

highlight_print(pipe._get_init_keys())
pdb.set_trace()

# implementation
latent_output = pipe(
    prompt_embeds=pos_prompt_embeds,
    negative_prompt_embeds=neg_prompt_embeds,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5,
    latents=latents,
    generator=generator,
    return_dict=False,
    output_type="latent",
)

latent = latent_output[0]
latent = latent / converted_vae.config.scaling_factor

image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
with torch.no_grad():
    image = converted_vae.decode(latent, return_dict=False, generator=generator)[0]
    image = image_processor.postprocess(image, output_type="pil")

save_dir = env.get_output_dir()

if batch_size == 1:
    output = image[0]
    output.save(os.path.join(save_dir, "output_sd15.png"))
elif batch_size > 1:
    for i, img in enumerate(image.images):
        save_path = os.path.join(save_dir, f"output_sd15_{i}.png")
        img.save(save_path)