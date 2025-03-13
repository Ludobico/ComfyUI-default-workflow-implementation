import torch
import psutil
from utils import highlight_print
from typing import Optional, Literal, Tuple

def get_torch_device():
    """
    It will be useful for all functions that use the `pytorch` framework
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device  

def get_cpu_device():
    """
    Return the CPU device if CUDA is not used.
    """
    device = 'cpu'
    return device

def get_memory_info(device_type : Literal['auto', 'cuda', 'cpu'] = 'auto', verbose : bool = True) -> Tuple[float, str]:
    """
    Return total VRAM (if CUDA is available) or total RAM (if using CPU)

    check your device the following code\n
    ```python
    from utils import get_torch_device

    device = get_torch_device()
    print(device)
    ```\n
    or check literal memory information

    ```python
    get_memory_info()
    get_memory_info('cuda')
    get_memory_info('cpu)
    ```
    """

    if device_type == 'auto' or device_type not in ('cuda', 'cpu'):
        device = get_torch_device()
    else:
        device = device_type

    if device == 'cuda':
        memory_type = "VRAM"
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        
    else:
        memory_type = "RAM"
        total_memory = psutil.virtual_memory().total / (1024**2)
        
    
    if verbose:
        highlight_print(f"Total {memory_type} : {total_memory:.0f} MB", color='green')

    return total_memory, memory_type
    