from typing import Optional,Literal, List, Union, Tuple
import torch
import psutil
from utils import highlight_print

def create_seed_generators(
        batch_size : int,
        seed : Optional[int] = None,
        task : Literal['fixed', 'increment', 'decrement', 'randomize'] = 'randomize',
        device = 'cpu'
) -> Union[torch.Generator, List[torch.Generator]]:
    """
    Generate a torch.Generator through task using the params `seed number` & `batch_size`
    """
    if device == 'cpu':
        device = get_cpu_device()
    else:
        device = get_torch_device()

    if batch_size < 1:
        raise ValueError("batch_size must be up to 1")
    
    SEED_MAX = 2**63
    if seed is None:
        seed = torch.randint(0, SEED_MAX - 1, (1,), device=device, dtype=torch.long).item()
    
    if task == 'fixed':
        generator = torch.Generator(device=device).manual_seed(seed)
        return generator
    
    elif task in ["increment", "decrement", "randomize"]:
        generators = []
        for i in range(batch_size):
            if task == 'increment':
                current_seed = seed + i
            elif task == 'decrement':
                current_seed = seed - i
            elif task == 'randomize':
                current_seed = torch.randint(0, SEED_MAX - 1, (1,), device=device, dtype=torch.long).item()

            current_seed = min(max(current_seed, 0), SEED_MAX)
            generator = torch.Generator(device=device).manual_seed(current_seed)
            generators.append(generator)
    return generators

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
    