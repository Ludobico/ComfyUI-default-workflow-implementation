import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from typing import Tuple, Optional
from module.module_utils import load_tokenizer
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin

class PromptEncoder(TextualInversionLoaderMixin):
    def sdxl_text_conditioning(
            self,
            prompt : str,
            clip : Tuple[CLIPTextModel, CLIPTextModelWithProjection],
            dtype : torch.dtype = torch.float16,
            clip_skip : Optional[int] = None
    ):
        tokenizer1, tokenizer2 = load_tokenizer("sdxl")

        text_encoder1 = clip[0].to(dtype=dtype)
        text_encoder2 = clip[1].to(dtype=dtype)
        device = next(text_encoder1.parameters()).device

        prompt2 = prompt
        prompt2 = [prompt2] if isinstance(prompt2, str) else prompt

        prompt_embeds_list = []
        prompts = [prompt, prompt2]
        tokenizers = [tokenizer1, tokenizer2]
        text_encoders = [text_encoder1, text_encoder2]

        for prompt, tokenizer, text_encoder in zip(prompts ,tokenizers, text_encoders):
            prompt = self.maybe_convert_prompt(prompt, tokenizer)
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length = tokenizer.model_max_length,
                truncation = True,
                return_tensors = "pt"
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )
            with torch.no_grad():
                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return (prompt_embeds, pooled_prompt_embeds)

    def sd15_text_conditioning(
            self,
            prompt : str,
            clip : CLIPTextModel,
            dtype : torch.dtype = torch.float16,
            clip_skip : Optional[int] = None
    ):
        tokenizer = load_tokenizer("sd15")
        text_encoder = clip.to(dtype=dtype)
        device = next(text_encoder.parameters()).device

        prompt = self.maybe_convert_prompt(prompt, tokenizer)

        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors='pt').input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        
        with torch.no_grad():
            if clip_skip is None:
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask = attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                # Skip the output of Transformer layers
                # if the parameter `clip_skip` is set to k. CLIP model's output will be (final_layer - k)
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask = attention_mask, output_hidden_states = True)
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # normalization
                prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)
        
        prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        # bs_embed, seq_len, _ = prompt_embeds.shape
        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return prompt_embeds
        

def sdxl_clip_postprocess(prompt_embeds, pooled_prompt_embeds, batch_size):
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, batch_size, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * batch_size, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, batch_size).view(bs_embed * batch_size, -1)
    return (prompt_embeds, pooled_prompt_embeds)

def sd_clip_postprocess(prompt_embeds, batch_size):
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, batch_size, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * batch_size, seq_len, -1)
    return prompt_embeds

