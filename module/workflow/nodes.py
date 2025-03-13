import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path
from config.getenv import GetEnv
from module.module_utils import limit_vram_usage
from module.model_state import extract_model_components
from module.model_architecture import UNet, VAE, TextEncoder
from module.converter.conversion import convert_unet_from_ckpt_sd, convert_vae_from_ckpt_sd, convert_clip_from_ckpt_sd
from typing import Union

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



def CLIP_text_encode(prompt : str):
    """
    The CLIPTextEncode node is designed to encode textual inputs using a CLIP model, transforming text into a form that can be utilized for conditioning in generative tasks. It abstracts the complexity of text tokenization and encoding, providing a streamlined interface for generating text-based conditioning vectors.
    """
    return prompt




if __name__ == "__main__":
    model_path = r"E:\st002\repo\generative\image\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"
    load_checkpoint(model_path)