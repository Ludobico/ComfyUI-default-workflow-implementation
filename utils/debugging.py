from functools import wraps
import time
from typing import Literal

def measure_time(print_lang : Literal['ko', 'en'] = 'ko'):
    """
    Measure execution time for a function or class using python decorator \n
    ```python
    @measure_time()
    async def foo():
        pass
    ```
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            if print_lang == 'ko':
                print(f"응답 시간 : {execution_time:.2f}초")

            elif print_lang == 'en':
                print(f"Execution time : {execution_time:.2f} seconds")
            
            else:
                print(f"Execution time : {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator

