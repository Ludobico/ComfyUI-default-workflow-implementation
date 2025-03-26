from nodes import load_checkpoint, CLIP_text_encode, empty_latent_image, k_sampler, vae_decode, save_image

model_path = r""

model, clip, vae = load_checkpoint(ckpt_name=model_path)

positive = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
negative = "text, watermark"

positive_conditioning = CLIP_text_encode(text=positive, clip=clip)
negative_conditioning = CLIP_text_encode(text=negative, clip=clip)

latent = empty_latent_image(width=1024, height=1024, batch_size=1)

LATENT = k_sampler(model=model, positive=positive_conditioning, negative=negative_conditioning, latent_image=latent, control_after_generate='randomize', steps=25, cfg=7.5, sampler_name='euler', scheduler='normal', denoise=1.00)

image = vae_decode(samples=LATENT, vae=vae)

save_image(images=image)