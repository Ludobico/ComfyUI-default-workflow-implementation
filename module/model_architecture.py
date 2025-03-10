from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTextModelWithProjection
import torch

class UNet:
    @staticmethod
    def sdxl():
        unet = UNet2DConditionModel(
            sample_size=128,
            act_fn="silu",
            addition_embed_type="text_time",
            addition_embed_type_num_heads=64,
            addition_time_embed_dim=256,
            attention_head_dim=[5,10,20],
            attention_type="default",
            block_out_channels=[320, 640, 1280],
            center_input_sample=False,
            class_embed_type=None,
            class_embeddings_concat=False,
            conv_in_kernel=3,
            conv_out_kernel=3,
            cross_attention_dim=2048,
            cross_attention_norm=None,
            down_block_types=["DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"],
            downsample_padding=1,
            dropout=0.0,
            dual_cross_attention=False,
            encoder_hid_dim=None,
            encoder_hid_dim_type=None,
            flip_sin_to_cos=True,
            freq_shift=0,
            in_channels=4,
            layers_per_block=2,
            mid_block_only_cross_attention=None,
            mid_block_scale_factor=1,
            mid_block_type="UNetMidBlock2DCrossAttn",
            norm_eps=1e-05,
            norm_num_groups=32,
            num_attention_heads=None,
            num_class_embeds=None,
            only_cross_attention=False,
            out_channels=4,
            projection_class_embeddings_input_dim=2816,
            resnet_out_scale_factor=1.0,
            resnet_skip_time_act=False,
            resnet_time_scale_shift="default",
            reverse_transformer_layers_per_block=None,
            time_cond_proj_dim=None,
            time_embedding_act_fn=None,
            time_embedding_type="positional",
            timestep_post_act=None,
            transformer_layers_per_block=[1,2,10],
            up_block_types=["CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"],
            upcast_attention=None,
            use_linear_projection=True
        )
        return unet
    

class VAE:
    @staticmethod
    def sdxl():
        """
        stabilityai/sdxl-vae
        """
        vae = AutoencoderKL(
            act_fn="silu",
            block_out_channels=[128,256,512,512],
            down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
            ],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=1024,
            scaling_factor=0.13025,
            up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
            ]
        )
        return vae
    
class TextEncoder:
    @staticmethod
    def sdxl_enc1():
        config = CLIPTextConfig(
            attention_dropout=0.0,
            bos_token_id=0,
            eos_token_id=2,
            dropout=0.0,
            hidden_act="quick_gelu",
            hidden_size=768,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=1,
            projection_dim=768,
            torch_dtype = torch.float16,
            vocab_size=49408
        )
        enc1 = CLIPTextModel(config=config)
        return enc1
    
    @staticmethod
    def sdxl_enc2_config():
        config = CLIPTextConfig(
            attention_dropout=0.0,
            bos_token_id=0,
            dropout = 0.0,
            eos_token_id= 2,
            hidden_act="gelu",
            hidden_size=1280,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=5120,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            num_attention_heads=20,
            num_hidden_layers=32,
            pad_token_id=1,
            projection_dim=1280,
            torch_dtype=torch.float16,
            vocab_size=49408
        )
        return config