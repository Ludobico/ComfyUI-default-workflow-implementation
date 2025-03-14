import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from typing import Literal, Tuple
from module.module_utils import load_tokenizer
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
import pdb
from utils import highlight_print

class PromptEncoder(TextualInversionLoaderMixin):
    def sdxl_text_conditioning(
            self,
            prompt : str,
            clip : Tuple[CLIPTextModel, CLIPTextModelWithProjection],
            dtype : torch.dtype = torch.float16
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
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return (prompt_embeds, pooled_prompt_embeds)




