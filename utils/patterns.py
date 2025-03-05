import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import re
from utils import highlight_print
from typing import Optional, Union, List

def validation_regex_on_requirements(file_path : os.PathLike, libraries_to_remove : Optional[Union[str, List[str]]] = None):
    """
    print `SED` command to use in dockerfile if want to remove specific libraries like torch
    """

    if libraries_to_remove is None:
        libraries_to_remove = ['torch', 'torchaudio', 'torchvision']
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file at : {file_path}")
    
    if isinstance(libraries_to_remove, str):
        libraries_to_remove = [libraries_to_remove]
    
    with open(file_path, 'r') as f:
        content = f.readlines()

    sed_patterns = []
    removed_lines = []

    for lib in libraries_to_remove:
        pattern = f"^{lib}(==.*)?$"
        regex = re.compile(pattern)
        
        for line in content:
            line = line.strip()
            if regex.match(line):
                removed_lines.append(line)
        
        sed_pattern = f"/^{lib}\\(\\|==.*\\)$/d"
        sed_patterns.append(sed_pattern)

    if removed_lines:
        for line in removed_lines:
            print(f" - {line}")
    else:
        print(f"Not found library which will be removed")
    
    sed_command = "RUN sed -i '" + "; ".join(sed_patterns) + f"' {os.path.basename(file_path)}"
    highlight_print(sed_command, color='green')

if __name__ == "__main__":
    req_path = os.path.join(project_root, 'requirements.txt')
    validation_regex_on_requirements(req_path)