from config.getenv import GetEnv
from utils import highlight_print, get_memory_info
from module.model_state import get_model_keys, extract_model_components
from module.module_utils import auto_model_detection
import os
from module.model_architecture import UNet
env = GetEnv()

vram_info, _ = get_memory_info(verbose=False)

print(_)
print(vram_info)