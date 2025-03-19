import os
from typing import Literal, Optional, Dict
from safetensors.torch import load_file
from utils import highlight_print
from module.torch_utils import get_cpu_device, get_torch_device, get_memory_info
import torch
from config.getenv import GetEnv
from diffusers import UNet2DConditionModel
from transformers import CLIPTokenizer
import json

env = GetEnv()

def load_checkpoint_file(ckpt, device : Literal['auto', 'gpu', 'cpu'] = 'auto'):
    """
    Load a model file (either .safetensors or .ckpt).
    """
    if device == 'auto' or device == 'gpu':
        device = get_torch_device()
    elif device == 'cpu':
        device = get_cpu_device()
    else:
        device = get_torch_device()
    
    ext = os.path.splitext(ckpt)[-1].lower()

    if ext == '.safetensors':
        sd = load_file(ckpt, device=device)
    elif ext == '.ckpt':
        model = torch.load(ckpt, map_location=device)
        if "state_dict" in model:
            sd = model['state_dict']
        else:
            sd = model
    else:
        raise ValueError(f"{os.path.basename(ckpt)} is not a `safetensors` or `ckpt`")
    return sd

def auto_model_detection(ckpt) -> str:
    """
    Is it a stable diffusion? or SDXL?\n

    ```python
    model = "path/to/stable_diffusion_v1-5.ckpt"
    model_type = auto_model_detection(model)
    print(model_type) # 'sd15'

    model = "path/to/sd_xl_base_1.safetensors"
    model_type = auto_model_detection(model)
    print(model_type) # 'sdxl'
    ```
    """
    from module.model_state import get_model_keys
    sd = get_model_keys(ckpt, return_type='list', save_as_file=False)

    prefix_sd = [[part for part in item.split('.')[:1]] for item in sd]
    
    def flatten(lst):
        result = []
        for item in lst:
            result.extend(item)
        return result
    
    # Flatten a nested lits into a single list
    preprocess_sd1 = flatten(prefix_sd)
    # Remove duplicates from list
    preprocess_sd2 = list(set(preprocess_sd1))
    
    if 'cond_stage_model' in preprocess_sd2:
        return 'sd15'
    
    if 'conditioner' in preprocess_sd2:
        return 'sdxl'
    
    else:
        raise ValueError("Cannot determine model type : No clear CLIP keys found in the state_dict.")


def compare_unet_models(model1: UNet2DConditionModel, model2: UNet2DConditionModel):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"{key} is missing in model2")
            return False
        if not torch.allclose(state_dict1[key], state_dict2[key], atol=1e-6):
            print(f"{key} values are different")
            return False

    print("Both models are identical")
    return True

def load_tokenizer(model_type : Literal['sd15', 'sdxl']):
    sd_tok1_dir = os.path.join(env.get_tokenizer_dir(), 'sd15_tokenizer')
    xl_tok1_dir = os.path.join(env.get_tokenizer_dir(), 'sdxl_tokenizer')
    xl_tok2_dir = os.path.join(env.get_tokenizer_dir(), 'sdxl_tokenizer_2')
    cache_dir = env.get_tokenizer_dir()
    if model_type == 'sd15':
        tokenizer = CLIPTokenizer.from_pretrained(sd_tok1_dir, cache_dir=cache_dir)
        return tokenizer
    elif model_type == 'sdxl':
        tokenizer1 = CLIPTokenizer.from_pretrained(xl_tok1_dir, cache_dir=cache_dir)
        tokenizer2 = CLIPTokenizer.from_pretrained(xl_tok2_dir, cache_dir=cache_dir)
        return (tokenizer1, tokenizer2)
    
def limit_vram_usage(device, max_vram_fraction = 0.5):
    
    vram_info, _ = get_memory_info(verbose=False)
    # 8gb vram
    if vram_info < float(8000):
        max_vram_fraction = 0.9
    if isinstance(device, str):
        if device == 'cuda':
            device = "cuda:0"
    torch.cuda.set_per_process_memory_fraction(max_vram_fraction, device=device)

def save_config_files(pipe, unet : bool = True, vae : bool = True, text_encoder : bool = True, tokenizer : bool = True ,suffix : Optional[str] = None):

    output_dir = env.get_output_dir()

    if unet:
        unet_config = pipe.unet.config
        save_unet_path = os.path.join(output_dir, "unet_config.json" if suffix is None else f"unet_config_{suffix}.json")
        with open(save_unet_path, 'w', encoding='utf-8') as f:
            json.dump(unet_config, f, indent=4)
    
    if vae:
        vae_config = pipe.vae.config
        save_vae_path = os.path.join(output_dir, "vae_config.json" if suffix is None else f"vae_config_{suffix}.json")
        with open(save_vae_path, 'w', encoding='utf-8') as f:
            json.dump(vae_config, f, indent=4)

    if text_encoder:
        text_encoder_config = pipe.text_encoder.config
        save_encoder_path = os.path.join(output_dir, "encoder_config.json" if suffix is None else f"encoder_config_{suffix}.json")
        with open(save_encoder_path, "w", encoding='utf-8') as f:
            json.dump(text_encoder_config, f, indent=4)
        
        if hasattr(pipe, 'text_encoder_2'):
            text_encoder2_config = pipe.text_encoder_2.config
            save_encoder2_path = os.path.join(output_dir, 'encoder2_config.json' if suffix is None else f"encoder2_config+{suffix}.json")
            with open(save_encoder2_path, 'w', encoding='utf-8') as f:
                json.dump(text_encoder2_config, f, indent=4)
    
    if tokenizer:
        tok1_config = pipe.tokenizer.config
        save_tok1_path = os.path.join(output_dir, 'tokenizer1_config.json' if suffix is None else f"tokenizer1_config_{suffix}.json")
        with open(save_tok1_path, 'w', encoding='utf-8') as f:
            json.dump(tok1_config, f, indent=4)
        
        if hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2:
            tok2_config = pipe.tokenizer_2.config
            save_tok2_path = os.path.join(output_dir, 'tokenizer2_config.json' if suffix is None else f"tokenizer2_config_{suffix}.json")
            with open(save_tok2_path, 'w', encoding='utf-8') as f:
                json.dump(tok2_config, f, indent=4)


    
    highlight_print("DONE", 'green')