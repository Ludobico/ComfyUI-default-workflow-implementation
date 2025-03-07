from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, SchedulerMixin
from module.sampler import common as CONFIG
from typing import Literal

eular = EulerDiscreteScheduler(**CONFIG.eular)
euler_ancestral = EulerAncestralDiscreteScheduler(**CONFIG.eular_a)
heun = HeunDiscreteScheduler(**CONFIG.heun)
dpm_2=KDPM2DiscreteScheduler(**CONFIG.dpm_2)
dpm_2_ancestral=KDPM2AncestralDiscreteScheduler(**CONFIG.dpm_2_ancestral)
lms=LMSDiscreteScheduler(**CONFIG.lms)
dpmpp_2m=DPMSolverMultistepScheduler(**CONFIG.dpmpp_2m)
dpmpp_2m_sde=DPMSolverMultistepScheduler(**CONFIG.dpmpp_2m, algorithm_type="sde-dpmsolver++")

def schedular_type(sampler_name  ,noise : Literal['normal', 'karras', 'sgm_uniform', 'simple', 'exponential', 'beta']):
    config_with_noise = dict(sampler_name.config)
    if noise == 'normal':
        return sampler_name
    elif noise == 'karras':
        config_with_noise['use_karras_sigmas'] = True
    elif noise == 'sgm_uniform':
        config_with_noise["timestep_spacing"] = "trailing"
    elif noise == 'simple':
        config_with_noise["timestep_spacing"] = "trailing"
    elif noise == 'exponential':
        config_with_noise["timestep_spacing"] = "linspace"
    elif noise == 'beta':
        config_with_noise['use_karras_sigmas'] = True
        config_with_noise["timestep_spacing"] = "linspace"


    if isinstance(sampler_name, EulerDiscreteScheduler):
        return EulerDiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, EulerAncestralDiscreteScheduler):
        return EulerAncestralDiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, HeunDiscreteScheduler):
        return HeunDiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, KDPM2DiscreteScheduler):
        return KDPM2DiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, KDPM2AncestralDiscreteScheduler):
        return KDPM2AncestralDiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, LMSDiscreteScheduler):
        return LMSDiscreteScheduler(**config_with_noise)
    elif isinstance(sampler_name, DPMSolverMultistepScheduler):
        return DPMSolverMultistepScheduler(**config_with_noise, algorithm_type="sde-dpmsolver++")
    else:
        raise ValueError(f"Unsupported sampler : {sampler_name}")