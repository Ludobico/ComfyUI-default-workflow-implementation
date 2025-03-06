# Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the UNet, VAE, and Text Encoder.
# Does not convert optimizer state or any other thing.

import os
import re
from typing import Dict

import torch

from config.getenv import GetEnv
from module.model_architecture import Unet

env = GetEnv()

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (HF Diffusers, stable-diffusion)
    ("time_embedding.linear_1.weight", "time_embed.0.weight"),
    ("time_embedding.linear_1.bias", "time_embed.0.bias"),
    ("time_embedding.linear_2.weight", "time_embed.2.weight"),
    ("time_embedding.linear_2.bias", "time_embed.2.bias"),
    ("conv_in.weight", "input_blocks.0.0.weight"),
    ("conv_in.bias", "input_blocks.0.0.bias"),
    ("conv_norm_out.weight", "out.0.weight"),
    ("conv_norm_out.bias", "out.0.bias"),
    ("conv_out.weight", "out.2.weight"),
    ("conv_out.bias", "out.2.bias"),
    # the following are for sdxl
    ("add_embedding.linear_1.weight", "label_emb.0.0.weight"),
    ("add_embedding.linear_1.bias", "label_emb.0.0.bias"),
    ("add_embedding.linear_2.weight", "label_emb.0.2.weight"),
    ("add_embedding.linear_2.bias", "label_emb.0.2.bias"),
]

unet_conversion_map_resnet = [
    # (HF Diffusers, stable-diffusion)
    ("norm1", "in_layers.0"),
    ("conv1", "in_layers.2"),
    ("norm2", "out_layers.0"),
    ("conv2", "out_layers.3"),
    ("time_emb_proj", "emb_layers.1"),
    ("conv_shortcut", "skip_connection"),
]

unet_conversion_map_layer = []
for i in range(3):
    for j in range(2):
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((hf_down_res_prefix, sd_down_res_prefix))  # Changed order

        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((hf_down_atn_prefix, sd_down_atn_prefix))  # Changed order

    for j in range(4):
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((hf_up_res_prefix, sd_up_res_prefix))  # Changed order

        if i < 2:
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((hf_up_atn_prefix, sd_up_atn_prefix))  # Changed order

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((hf_downsample_prefix, sd_downsample_prefix))  # Changed order

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((hf_upsample_prefix, sd_upsample_prefix))  # Changed order

unet_conversion_map_layer.append(("output_blocks.2.1.conv.", "output_blocks.2.2.conv."))  # Changed order

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((hf_mid_atn_prefix, sd_mid_atn_prefix))  # Changed order

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((hf_mid_res_prefix, sd_mid_res_prefix))  # Changed order

# def convert_unet_state_dict(unet_state_dict):
#     # buyer beware: this is a *brittle* function,
#     # and correct output requires that all of these pieces interact in
#     # the exact order in which I have arranged them.
#     mapping = {k: k for k in unet_state_dict.keys()}
#     for sd_name, hf_name in unet_conversion_map:
#         mapping[hf_name] = sd_name
#     for k, v in mapping.items():
#         if "resnets" in k:
#             for sd_part, hf_part in unet_conversion_map_resnet:
#                 v = v.replace(hf_part, sd_part)
#             mapping[k] = v
#     for k, v in mapping.items():
#         for sd_part, hf_part in unet_conversion_map_layer:
#             v = v.replace(hf_part, sd_part)
#         mapping[k] = v
#     new_state_dict = {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}
#     return new_state_dict

with open(os.path.join(env.get_output_dir(), 'unet_conversion_map_layer.txt'), 'w', encoding='utf-8') as f:
    for item in unet_conversion_map_layer:
        f.write(str(item) + '\n')  # 튜플을 문자열로 변환 후 저장
def convert_unet_state_dict(unet_state_dict : Dict):
    hf_unet = Unet.sdxl()
    hf_unet_sd = hf_unet.state_dict()
    mapping = {k : k for k in hf_unet_sd.keys()}

    for hf_name, sd_name in unet_conversion_map:
        mapping[sd_name] = hf_name
    
    for k, v in mapping.items():
        if "resnets" in k:
            for hf_part, sd_part in unet_conversion_map_resnet:
                v = v.replace(sd_part, hf_part)
            mapping[k] = v
    
    for k, v in mapping.items():
        for hf_part, sd_part in unet_conversion_map_layer:
            v = v.replace(sd_part, hf_part)
        mapping[k] = v
    
    new_state_dict = {hf_name : unet_state_dict[sd_name] for sd_name, hf_name in mapping.items()}
    return new_state_dict