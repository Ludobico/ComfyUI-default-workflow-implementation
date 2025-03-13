import os
from utils import highlight_print

class GetEnv:
    def __init__(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))
    
    def get_project_dir(self):
        project_dir = os.path.abspath(os.path.join(self.curdir, '..'))
        return project_dir
    def get_model_dir(self):
        model_dir = os.path.join(self.curdir, '..' ,'models')
        return model_dir
    
    def get_ckpt_dir(self):
        ckpt_dir = os.path.join(self.get_model_dir(), 'checkpoints')
        return ckpt_dir
    
    def get_clip_model_dir(self):
        clip_model_dir = os.path.join(self.get_model_dir(), 'CLIP')
        return clip_model_dir
    
    def get_checkpoint_model_dir(self):
        checkpoint_model_dir = os.path.join(self.get_model_dir(), 'checkpoints')
        return checkpoint_model_dir
    
    def get_output_dir(self):
        output_dir = os.path.join(self.get_project_dir(), 'output')
        return output_dir
    
    def get_tokenizer_dir(self):
        tokenizer_dir = os.path.join(self.get_model_dir(), 'tokenizer')
        return tokenizer_dir