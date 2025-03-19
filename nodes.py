import os, sys
from pathlib import Path
from config.getenv import GetEnv
from module.module_utils import limit_vram_usage
from module.model_state import extract_model_components
from module.model_architecture import UNet, VAE, TextEncoder
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from typing import Union
from module.encoder import PromptEncoder

env = GetEnv()

MODEL_TYPE = ""

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

        model = convert_unet_from_ckpt_sd(original_unet, ckpt_model)
        vae = convert_vae_from_ckpt_sd(original_vae, ckpt_vae)
        enc1, enc2 = convert_clip_from_ckpt_sd(original_encoder, ckpt_clip, model_type)
        clip = (enc1, enc2)
        return model, clip, vae
    
    elif model_type == 'sd15':
        original_unet = UNet.sd15()
        original_vae = VAE.sd15()
        original_encoder = TextEncoder.sd15_enc()
        
        model = convert_clip_from_ckpt_sd(original_unet, ckpt_model)
        vae = convert_vae_from_ckpt_sd(original_vae, ckpt_vae)
        clip = convert_clip_from_ckpt_sd(original_encoder, ckpt_clip, model_type)
        return model, clip, vae


def CLIP_text_encode(clip, text : str):
    """
    The CLIPTextEncode node is designed to encode textual inputs using a CLIP model, transforming text into a form that can be utilized for conditioning in generative tasks. It abstracts the complexity of text tokenization and encoding, providing a streamlined interface for generating text-based conditioning vectors.

    ## Input types
    text : The `text` parameter is the textual input that will be encoded. It plays a crucial role in determining the output conditioning vector, as it is the primary source of information for the encoding process

    clip : The `clip` parameter represents the CLIP model used for tet tokenization and encoding. It is essential for converting the textual into a conditioning vector. influencing the quality and relevance of the generated output.

    ## Output types
    conditioning : The output `conditioning` is a vector representation of the input text, encoded by the CLIP model. It serves as a crucial component for guiding generative models in producing relevant and coherent outputs.
    """
    encoder = PromptEncoder()
    if MODEL_TYPE == 'sdxl':
        prompt_embeds, pooled_prompt_embeds = encoder.sdxl_text_conditioning(prompt=text, clip=clip)
        conditioning = (prompt_embeds, pooled_prompt_embeds)
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
    pass


if __name__ == "__main__":
    model_path = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"
    load_checkpoint(model_path)