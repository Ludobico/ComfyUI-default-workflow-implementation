import os
from utils import highlight_print

class GetEnv:
    def __init__(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))
    
    def get_model_dir(self):
        model_dir = os.path.join(self.curdir, '..' ,'models')
        return model_dir
    
    def get_clip_model_dir(self):
        clip_model_dir = os.path.join(self.get_model_dir(), 'CLIP')
        return clip_model_dir
    
    def get_checkpoint_model_dir(self):
        checkpoint_model_dir = os.path.join(self.get_model_dir(), 'checkpoints')
        return checkpoint_model_dir