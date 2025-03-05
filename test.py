from config.getenv import GetEnv
from utils import highlight_print

env = GetEnv()

highlight_print(env.get_clip_model_dir())