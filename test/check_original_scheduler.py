from diffusers import EulerDiscreteScheduler

SD_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_name = "stabilityai/stable-diffusion-xl-base-1.0"

scheduler = EulerDiscreteScheduler.from_pretrained(SD_name, subfolder="scheduler")

print(scheduler.config)

KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
                  "gradient_estimation"]

# SCHEDULER_HANDLERS = {
#     "normal": SchedulerHandler(normal_scheduler),
#     "karras": SchedulerHandler(k_diffusion_sampling.get_sigmas_karras, use_ms=False),
#     "exponential": SchedulerHandler(k_diffusion_sampling.get_sigmas_exponential, use_ms=False),
#     "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
#     "simple": SchedulerHandler(simple_scheduler),
#     "ddim_uniform": SchedulerHandler(ddim_scheduler),
#     "beta": SchedulerHandler(beta_scheduler),
#     "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
#     "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
# }