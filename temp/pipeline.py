from diffusers import StableDiffusionXLPipeline
import torch
import os
from module.module_utils import get_torch_device, limit_vram_usage
from config.getenv import GetEnv

env = GetEnv()
def pipeline_for_test(ckpt):
    prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt = "text, watermark"
    device = get_torch_device()
    limit_vram_usage(device=device)
    pipe = StableDiffusionXLPipeline.from_single_file(ckpt)

    pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=1024,
        width=1024
    )
    output = image.images[0]
    save_path = env.get_output_dir()

    output.save(os.path.join(save_path, 'output.png'))
    print("DONE")