from config.getenv import GetEnv
from utils import highlight_print, get_memory_info
from module.model_state import get_model_keys, extract_model_components
from module.module_utils import auto_model_detection
import os
from module.model_architecture import UNet
from module.torch_utils import create_seed_generators

batch_size = 5

rand_gen = create_seed_generators(5, task='fixed')

for gen in rand_gen:
    highlight_print(gen.initial_seed(), 'blue')