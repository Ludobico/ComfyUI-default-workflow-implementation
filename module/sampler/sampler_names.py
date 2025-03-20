from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, SchedulerMixin
from module.sampler import common as CONFIG
from typing import Literal

SAMPLER_MAP = {
    'euler': (EulerDiscreteScheduler, CONFIG.euler),
    'euler_ancestral': (EulerAncestralDiscreteScheduler, CONFIG.eular_a),
    'heun': (HeunDiscreteScheduler, CONFIG.heun),
    'dpm_2': (KDPM2DiscreteScheduler, CONFIG.dpm_2),
    'dpm_2_ancestral': (KDPM2AncestralDiscreteScheduler, CONFIG.dpm_2_ancestral),
    'lms': (LMSDiscreteScheduler, CONFIG.lms),
    'dpmpp_2m': (DPMSolverMultistepScheduler, CONFIG.dpmpp_2m),
    'dpmpp_2m_sde': (DPMSolverMultistepScheduler, CONFIG.dpmpp_2m_sde),
}

NOISE_CONFIG = {
    'normal': {},
    'karras': {'use_karras_sigmas': True},
    'sgm_uniform': {'timestep_spacing': 'trailing'},
    'simple': {'timestep_spacing': 'trailing'},
    'exponential': {'timestep_spacing': 'linspace'},
    'beta': {'use_karras_sigmas': True, 'timestep_spacing': 'linspace'},
}


def scheduler_type(sampler_name : Literal['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpmpp_2m', 'dpmpp_2m_sde'],
                   noise : Literal['normal', 'karras', 'sgm_uniform', 'simple', 'exponential', 'beta']):

    if sampler_name not in SAMPLER_MAP:
            raise ValueError(f"Unsupported sampler: {sampler_name}")

    scheduler_class, base_config = SAMPLER_MAP[sampler_name]
    config = {**base_config, **NOISE_CONFIG.get(noise, {})}

    return scheduler_class(**config)