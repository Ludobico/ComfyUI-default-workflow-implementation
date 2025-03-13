import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Literal, Tuple
from module.module_utils import load_tokenizer

def sdxl_text_conditioning(
        prompt : str,
        clip : Tuple,
        device : Literal['gpt','cpu','auto'] = 'auto',
        dtype : torch.dtype = torch.float16
):
    tokenizer1, tokenizer2 = load_tokenizer("sdxl")
    text_encoder1 = clip[0]
    text_encoder2 = clip[1]

    tokens_1 = tokenizer1(
        [prompt],
        padding="max_length",
        max_length=tokenizer1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


    tokens_2 = tokenizer2(
        [prompt],
        padding="max_length",
        max_length=tokenizer2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    print(type(tokens_1))
    print(type(tokens_2))

    with torch.no_grad():
        emb_1 = text_encoder1(tokens_1)[0]
        emb_2 = text_encoder2(tokens_2)[0]

    embedded_prompt = torch.cat([emb_1, emb_2], dim=1)
    return embedded_prompt
