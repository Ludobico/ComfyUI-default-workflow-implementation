import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import inspect
from typing import Callable, Any, Type, List
from utils.highlight import highlight_print

def function_help(func_or_class : Callable[[Any], Any]):
    help(func_or_class)
    highlight_print(func_or_class.__code__.co_varnames)

def function_inspect(func_or_class : Callable[[Any], Any]):
    """
    Inspect keyword arguments of a function or a class
    """

    parameters = inspect.signature(func_or_class).parameters
    for name, param in parameters.items():
        print(f"Parameter : {name}, kind : {param.kind}")

def class_methods_instpect(cls : Type) -> List[str]:
    """
    Inspect all methods of a class
    """

    if not inspect.isclass(cls):
        raise TypeError(f"{cls} is not a class")
    
    methods = []

    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        methods.append(name)
        print(f"Method : {name}, Defined in : {member.__module__}")
    
