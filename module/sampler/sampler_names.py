from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from module.sampler.common import config

Eular = EulerDiscreteScheduler(**config)
