from config.getenv import GetEnv
from utils import highlight_print
from module.module_utils import get_model_keys

env = GetEnv()

model_path = r"C:\Users\aqs45\OneDrive\Desktop\repo\ComfyUI-default-workflow-implementation\models\checkpoints\[PONY]prefectPonyXL_v50.safetensors"

get_model_keys(model_path, device='cpu')