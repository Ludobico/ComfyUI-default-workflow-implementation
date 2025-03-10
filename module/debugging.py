import os, sys, pdb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from diffusers import StableDiffusionXLPipeline
import torch
from module.module_utils import get_torch_device, limit_vram_usage, load_tokenizer, save_config_files
from config.getenv import GetEnv
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from module.sampler.sampler_names import euler_ancestral, scheduler_type
from module.model_state import  extract_model_components
from module.model_architecture import UNet, VAE, TextEncoder
from utils import highlight_print


ckpt = r""
env = GetEnv()
cache_dir = os.path.join(env.get_project_dir(), 'temp')

def pipe_from_diffusers():
    torch.cuda.empty_cache()
    prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt = "text, watermark"
    seed = 2025
    device = get_torch_device()
    limit_vram_usage(device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    pipe = StableDiffusionXLPipeline.from_single_file(ckpt, torch_dtype = torch.float16, cache_dir=cache_dir)
    return pipe

def pipe_self_made():
    unet = UNet.sdxl()
    vae = VAE.sdxl()
    enc1 = TextEncoder.sdxl_enc1()
    device = get_torch_device()
    limit_vram_usage(device=device)
    ckpt_unet_tensors, clip_tensors, vae_tensors, model_type = extract_model_components(ckpt)

    converted_unet = convert_unet_from_ckpt_sd(unet, ckpt_unet_tensors)
    converted_vae = convert_vae_from_ckpt_sd(vae, vae_tensors)
    converted_enc1, converted_enc2 = convert_clip_from_ckpt_sd(enc1, clip_tensors, model_type)

    prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt = "text, watermark"
    seed = 2025
    device = get_torch_device()
    limit_vram_usage(device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    tokenizer1, tokenizer2 = load_tokenizer(model_type)


    schedular = scheduler_type(euler_ancestral, 'normal')

    pipe = StableDiffusionXLPipeline(
        unet=converted_unet,
        vae=converted_vae,
        text_encoder=converted_enc1,
        text_encoder_2=converted_enc2,
        tokenizer=tokenizer1,
        tokenizer_2=tokenizer2,
        scheduler=schedular
        )
    return pipe

def compare_tensor_keys(diffuser_pipe : StableDiffusionXLPipeline, self_made_pipe : StableDiffusionXLPipeline):
    components = {
        "unet" : (diffuser_pipe.unet, self_made_pipe.unet),
        "vae" : (diffuser_pipe.vae, self_made_pipe.vae),
        "text_encoder": (diffuser_pipe.text_encoder, self_made_pipe.text_encoder),
        "text_encoder_2": (diffuser_pipe.text_encoder_2, self_made_pipe.text_encoder_2)
    }

    differences = {}

    for comp_name, (model1, model2) in components.items():
        # diffusers
        keys1 = set(model1.state_dict().keys())
        # self
        keys2 = set(model2.state_dict().keys())

        only_in_diffusers = keys1 - keys2
        only_in_made_self = keys2 - keys1

        differences[comp_name] = {
            "only_in_pipe1": list(only_in_diffusers),
            "only_in_pipe2": list(only_in_made_self),
            "has_difference": len(only_in_diffusers) > 0 or len(only_in_made_self) > 0
        }

    output_path = os.path.join(env.get_output_dir(), 'compare_keys.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(differences, f, ensure_ascii=False, indent=4)
    
    highlight_print("DONE", 'green')

if __name__ == "__main__":
    pipe1 = pipe_from_diffusers()
    pipe2 = pipe_self_made()

    save_config_files(pipe1, False, False, False)
    save_config_files(pipe2, False, False, False, suffix="test")