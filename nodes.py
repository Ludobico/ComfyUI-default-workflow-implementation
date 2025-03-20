import os, sys
import torch
from pathlib import Path
from config.getenv import GetEnv
from module.module_utils import limit_vram_usage, load_tokenizer
from module.model_state import extract_model_components
from module.model_architecture import UNet, VAE, TextEncoder
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from typing import Union, Literal, Optional
from module.encoder import PromptEncoder, sdxl_clip_postprocess, sd_clip_postprocess
from module.sampler.ksample_elements import retrieve_timesteps, prepare_latents
from module.sampler.sampler_names import scheduler_type
from module.torch_utils import create_seed_generators, get_torch_device
from diffusers import StableDiffusionXLPipeline

env = GetEnv()

MODEL_TYPE = ""
ENCODER = None

# All Documentation comes from the ComfyUI Wiki.
# See https://comfyui-wiki.com/en for more details.

def load_checkpoint(ckpt_name : Union[os.PathLike, str]):
    """
    The CheckpointLoaderSimple node is designed for loading model checkpoints without the need for specifying a configuration. It simplifies the process of checkpoint loading by requiring only the checkpoint name, making it more accessible for users who may not be familiar with the configuration details.

    ## Input types

    ckpt_name : Specifies the name of the checkpoint to be loaded, determining which checkpoint file the node will attempt to load and affecting the node’s execution and the model that is loaded.

    ## Output types
    model : Returns the loaded model, allowing it to be used for further processing or inference.

    clip : Returns the CLIP model associated with the loaded checkpoint, if available.

    vae : Returns the VAE model associated with the loaded checkpoint, if available.
    """
    global MODEL_TYPE

    if not os.path.isabs(ckpt_name):
        ckpt_dir = env.get_ckpt_dir()
        ckpt_name = os.path.join(ckpt_dir, ckpt_name)
    
    if not os.path.isfile(ckpt_name):
        raise FileNotFoundError(f"Checkpoint file (.bin or .safetensors) not found : {ckpt_name}")
    
    ckpt_model, ckpt_clip, ckpt_vae, model_type = extract_model_components(ckpt_name)
    MODEL_TYPE = model_type

    if model_type == 'sdxl':
        original_unet = UNet.sdxl()
        original_vae = VAE.sdxl()
        original_encoder = TextEncoder.sdxl_enc1()

        unet = convert_unet_from_ckpt_sd(original_unet, ckpt_model)
        vae = convert_vae_from_ckpt_sd(original_vae, ckpt_vae)
        enc1, enc2 = convert_clip_from_ckpt_sd(original_encoder, ckpt_clip, model_type)
        clip = (enc1, enc2)
        model = (unet, vae, clip)
        return model, clip, vae
    
    elif model_type == 'sd15':
        original_unet = UNet.sd15()
        original_vae = VAE.sd15()
        original_encoder = TextEncoder.sd15_enc()
        
        unet = convert_clip_from_ckpt_sd(original_unet, ckpt_model)
        vae = convert_vae_from_ckpt_sd(original_vae, ckpt_vae)
        clip = convert_clip_from_ckpt_sd(original_encoder, ckpt_clip, model_type)
        model = (unet, vae, clip)
        return model, clip, vae


def CLIP_text_encode(text : str, clip):
    """
    The CLIPTextEncode node is designed to encode textual inputs using a CLIP model, transforming text into a form that can be utilized for conditioning in generative tasks. It abstracts the complexity of text tokenization and encoding, providing a streamlined interface for generating text-based conditioning vectors.

    ## Input types

    text : The `text` parameter is the textual input that will be encoded. It plays a crucial role in determining the output conditioning vector, as it is the primary source of information for the encoding process

    clip : The `clip` parameter represents the CLIP model used for tet tokenization and encoding. It is essential for converting the textual into a conditioning vector. influencing the quality and relevance of the generated output.

    ## Output types
    conditioning : The output `conditioning` is a vector representation of the input text, encoded by the CLIP model. It serves as a crucial component for guiding generative models in producing relevant and coherent outputs.
    """
    global ENCODER
    ENCODER = PromptEncoder()
    if MODEL_TYPE == 'sdxl':
        prompt_embeds, pooled_prompt_embeds = ENCODER.sdxl_text_conditioning(prompt=text, clip=clip)
        conditioning = (prompt_embeds, pooled_prompt_embeds)
        return conditioning
    elif MODEL_TYPE == 'sd':
        prompt_embeds = ENCODER.sd15_text_conditioning(prompt=text, clip=clip)
        conditioning = prompt_embeds
        return conditioning
    


def empty_latent_image(width : int = 512, height : int = 512, batch_size : int = 1):
    """
    The EmptyLatentImage node is designed to generate a blank latent space representation with specified dimensions and batch size. This node serves as a foundational step in generating or manipulating images in latent space, providing a starting point for further image synthesis or modification processes.

    ## Input types

    width : Specifies the width of the latent image to be generated. This parameter directly influences the spatial dimensions of the resulting latent representation.

    height : Determines the height of the latent image to be generated. This parameter is crucial for defining the spatial dimensions of the latent space representation.

    batch_size : Controls the number of latent images to be generated in a single batch. This allows for the generation of multiple latent representations simultaneously, facilitating batch processing.

    ## Output types
    latent : The output is a tensor representing a batch of blank latent images, serving as a base for further image generation or manipulation in latent space.
    """
    latent = (height, width, batch_size)
    return latent

def k_sampler(model, positive, negative, latent_image,
              seed : Optional[int] = None,
              control_after_generate : Literal['fixed', 'increment', 'decrement', 'randomize'] = 'randomize',
              steps : int = 20,
              cfg : float = 8.0,
              sampler_name : Literal['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpmpp_2m', 'dpmpp_2m_sde'] = 'euler', scheduler : Literal['normal', 'karras', 'sgm_uniform', 'simple', 'exponential', 'beta'] = 'normal',
              denoise : float = 1.00):
    """
    The Ksampler node is designed for advanced sampling operations whihin generative models, allowing for the customization of sampling processes through various parameters. It facilitates the generation of new data samples by manipulating latent space representations, leveraging conditioning, and adjusting noise levels.

    ## Input types

    model : Specifies the generative model to be used for sampling, playing a crucial role in determining the characteristics of the generated samples.

    seed : Controls the randomness of the sampling process, ensuring reproducibility of results when set to a specific value.

    steps : Determines the number of steps to be taken in the sampling process, affecting the detail and quality of the generated samples.

    cfg : Adjusts the conditioning factor, influencing the direction and strength of the conditioning applied during sampling.

    sampler_name : Selects the specific sampling algorithm to be used, impacting the behavior and outcome of the sampling process.

    scheduler : Chooses the scheduling algorithm for controlling the sampling process, affecting the progression and dynamics of sampling.

    positive : Defines positive conditioning to guide the sampling towards desired attributes or features.

    negative : Specifies negative conditioning to steer the sampling away from certain attributes or features.

    latent_image : 	Provides a latent space representation to be used as a starting point or reference for the sampling process.

    denoise : Controls the level of denoising applied to the samples, affecting the clarity and sharpness of the generated images.

    ## Output types

    latent : Represents the latent space output of the sampling process, encapsulating the generated samples.
    """
    torch.cuda.empty_cache()
    env = GetEnv()
    dtype = torch.float16
    device = get_torch_device()

    limit_vram_usage(device=device)

    height, width, batch_size = latent_image

    if MODEL_TYPE == 'sdxl':
        # model = (unet, vae, clip)
        diffuser_scheduler = scheduler_type(sampler_name, scheduler)
        tokenizer1, tokenizer2 = load_tokenizer("sdxl")
        timesteps, num_inference_steps = retrieve_timesteps(diffuser_scheduler, num_inference_steps=steps, device=device)
        num_channels_latents = model[0].config.in_channels
        vae_scale_factor = 2 ** (len(model[1].config.block_out_channels) - 1)
        generator = create_seed_generators(batch_size, seed=seed, task=control_after_generate)

        positive, pooled_positive = sdxl_clip_postprocess(positive[0], positive[1])
        negative, pooled_negative = sdxl_clip_postprocess(negative[0], negative[1])

        latents = prepare_latents(batch_size, num_channels_latents, height, width, dtype, torch.device(device), generator, vae_scale_factor)

        pipe = StableDiffusionXLPipeline(
            unet=model[0],
            vae=model[1],
            text_encoder=model[2][0],
            text_encoder_2=model[2][1],
            tokenizer=tokenizer1,
            tokenizer_2=tokenizer2,
            scheduler=scheduler
        )



        





if __name__ == "__main__":
    model_path = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"
    load_checkpoint(model_path)