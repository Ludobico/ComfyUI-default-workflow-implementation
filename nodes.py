import os, pdb
import torch
from config.getenv import GetEnv
from module.module_utils import load_tokenizer, upcast_vae, get_save_image_path
from module.model_state import extract_model_components
from module.model_architecture import UNet, VAE, TextEncoder
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from typing import Union, Literal, Optional, Tuple, List
from module.encoder import PromptEncoder, sdxl_clip_postprocess, sd_clip_postprocess
from module.sampler.ksample_elements import retrieve_timesteps, prepare_latents
from module.sampler.sampler_names import scheduler_type
from module.torch_utils import create_seed_generators, get_torch_device, limit_vram_usage
from module.tensor_utils import randn_tensor
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from utils import highlight_print

env = GetEnv()

MODEL_TYPE = ""
ENCODER = None
num_channels_latents = None
vae_scale_factor = None
device = get_torch_device()
dtype = torch.float16

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
    global num_channels_latents
    global vae_scale_factor

    if not os.path.isabs(ckpt_name):
        ckpt_dir = env.get_ckpt_dir()
        ckpt_name = os.path.join(ckpt_dir, ckpt_name)
    
    if not os.path.isfile(ckpt_name):
        raise FileNotFoundError(f"Checkpoint file (.ckpt or .safetensors) not found : {ckpt_name}")
    
    ckpt_model, ckpt_clip, ckpt_vae, model_type = extract_model_components(ckpt_name)
    MODEL_TYPE = model_type

    if model_type == 'sdxl':
        original_unet = UNet.sdxl()
        num_channels_latents = original_unet.config.in_channels
        original_vae = VAE.sdxl()
        vae_scale_factor = 2 ** (len(original_vae.config.block_out_channels) - 1)
        original_encoder = TextEncoder.sdxl_enc1()

        unet = convert_unet_from_ckpt_sd(original_unet, ckpt_model)
        vae = convert_vae_from_ckpt_sd(original_vae, ckpt_vae)
        enc1, enc2 = convert_clip_from_ckpt_sd(original_encoder, ckpt_clip, model_type)
        clip = (enc1, enc2)
        model = (unet, vae, clip)
        return model, clip, vae
    
    elif model_type == 'sd15':
        original_unet = UNet.sd15()
        num_channels_latents = original_unet.config.in_channels
        original_vae = VAE.sd15()
        original_encoder = TextEncoder.sd15_enc()
        vae_scale_factor = 2 ** (len(original_vae.config.block_out_channels) - 1)
        
        unet = convert_unet_from_ckpt_sd(original_unet, ckpt_model)
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
    elif MODEL_TYPE == 'sd15':
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

    empty_latent = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    return empty_latent

def k_sampler(model : Tuple,
              positive : str,
              negative : str,
              latent_image : Union[Tuple, List],
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
    limit_vram_usage(device=device)

    generator = create_seed_generators(latent_image[0], seed=seed, task=control_after_generate)
    diffuser_scheduler = scheduler_type(sampler_name, scheduler)
    # Most schedulers do not support custom timesteps, so the default value is used in diffusers
    timesteps, num_inference_steps = retrieve_timesteps(diffuser_scheduler, num_inference_steps=steps, device=device)
    empty_latent = randn_tensor(shape=latent_image, generator=generator, device=torch.device(device), dtype=dtype)

    if MODEL_TYPE == 'sdxl':
        tokenizer1, tokenizer2 = load_tokenizer("sdxl")
        positive_embeds, pooled_positive_embeds = sdxl_clip_postprocess(positive[0], positive[1], batch_size=latent_image[0])
        negative_embeds, pooled_negative_embeds = sdxl_clip_postprocess(negative[0], negative[1], batch_size=latent_image[0])

        pipe = StableDiffusionXLPipeline(
            unet=model[0],
            vae=model[1],
            text_encoder=model[2][0],
            text_encoder_2=model[2][1],
            tokenizer=tokenizer1,
            tokenizer_2=tokenizer2,
            scheduler=diffuser_scheduler
        )
        pipe.to(device=device, dtype=dtype)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        latent_output = pipe(
            prompt_embeds=positive_embeds,
            pooled_prompt_embeds=pooled_positive_embeds,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=pooled_negative_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            latents=empty_latent,
            generator=generator,
            return_dict=False,
            output_type="latent",
            denoising_end=denoise
        )

        latent_output = latent_output[0]
        return latent_output
    
    if MODEL_TYPE == 'sd15':
        tokenizer1 = load_tokenizer('sd15')
        positive_embeds = sd_clip_postprocess(positive, batch_size=latent_image[0])
        negative_embeds = sd_clip_postprocess(negative, batch_size=latent_image[0])

        pipe = StableDiffusionPipeline(
            unet=model[0],
            vae=model[1],
            text_encoder=model[2],
            tokenizer=tokenizer1,
            scheduler=diffuser_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None
        )
        pipe.to(device=device, dtype=dtype)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        if denoise < 1.0 :
            highlight_print("Warning : Adjusting `denoise` reduces the number of steps. This only shortens the total steps proportionally and does not control the full denoising process precisely, unlike methods that adjust the entire timestep schedule.", 'green')
            num_inference_steps = round(num_inference_steps * denoise)
        latent_output = pipe(
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            latents=empty_latent,
            generator=generator,
            return_dict=False,
            output_type="latent"
        )

        latent_output = latent_output[0]
        return latent_output
    

def vae_decode(samples : torch.Tensor, vae : AutoencoderKL):
    """
    The VAEDecode node is designed for decoding latent representations into images using a specified Variational Autoencoder (VAE). It serves the purpose of generating images from compressed data representations, facilitating the reconstruction of images from their latent space encodings.

    ## Input types

    samples : The ‘samples’ parameter represents the latent representations to be decoded into images. It is crucial for the decoding process as it provides the compressed data from which the images are reconstructed.

    vae : The ‘vae’ parameter specifies the Variational Autoencoder model to be used for decoding the latent representations into images. It is essential for determining the decoding mechanism and the quality of the reconstructed images.

    ## Output types

    image : The output is an image reconstructed from the provided latent representation using the specified VAE model.
    """
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    if MODEL_TYPE == 'sdxl':
        needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast

        if needs_upcasting:
            vae = upcast_vae(vae)
            samples = samples.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        elif samples.dtype != vae.dtype:
            if torch.backends.mps.is_available():
                vae = vae.to(samples.dtype)
        
        has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
        has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(samples.device, samples.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(samples.device, samples.dtype)
            )
            samples = samples * latents_std / vae.config.scaling_factor + latents_mean
        else:
            samples = samples / vae.config.scaling_factor


        with torch.no_grad():
            image = vae.decode(samples, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")
        
        if needs_upcasting:
            vae.to(dtype=dtype)
        
        return image

    elif MODEL_TYPE == 'sd15':
        samples = samples / vae.config.scaling_factor

        with torch.no_grad():
            image = vae.decode(samples, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")
        
        return image


def save_image(images : List, filename_prefix : str = "ComfyUI"):
    """
    The Save Image node is mainly used to save images to the `output` folder. If you only want to preview the image during the intermediate process rather than saving it, you can use the `Preview Image` node. Default save location : `ComfyUI-default-workflow-implementation/output`

    ## Input types

    images : The images to be saved. This parameter is crucial as it directly contains the image data that will be processed and saved to disk.
    filename_prefix : The filename prefix for images saved to the ComfyUI-default-workflow-implementation/output/ folder. The default is ComfyUI, but you can customize it.
    """
    batch_size = len(images)
    full_output_folder, filename, counter, _, _ = get_save_image_path(filename_prefix)

    if batch_size == 1:
        output = images[0]
        save_path = os.path.join(full_output_folder, f"{filename}_{counter:03d}.png")
        output.save(save_path)
        print(f"Saved image : {save_path}")
    else:
        for i, img in enumerate(images):
            save_path = os.path.join(full_output_folder, f"{filename}_{counter + i:03d}.png")
            img.save(save_path)
            print(f"Saved image {i + 1}/{batch_size}: {save_path}")


def preview_image(images : List):
    """
    The previewImage node is designed for creating temporary preview images. It automatically generates a unique temporary file name for each image, compresses the image to a specified level, and saves it to a temporary directory. This functionality is particularly useful for generating previews of images during processing without affecting the original files.\n

    **However, this implementation takes a different approach.**\n

    This node is designed to display images directly using `Image.show()` without generating a temporary file name or saving to a temporary diretory.

    ## Input types

    images : The ‘images’ input specifies the images to be processed and saved as temporary preview images. This is the primary input for the node, determining which images will undergo the preview generation process.

    ## Output types

    The node doesn’t have output types.
    """
    batch_size = len(images)
    
    if batch_size == 1:
        output = images[0]
        output.show()
    else:
        for img in images:
            img.show()



        
