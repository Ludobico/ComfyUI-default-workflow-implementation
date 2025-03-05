from config.getenv import GetEnv
from utils import highlight_print
from module.model_state import get_model_keys
from module.module_utils import auto_model_detection

env = GetEnv()

model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"

auto_model_detection(model_path)