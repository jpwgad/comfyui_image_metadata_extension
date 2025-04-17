import glob
import importlib
import os
import sys

from .captures import CAPTURE_FIELD_LIST
from .samplers import SAMPLERS

# load CAPTURE_FIELD_LIST and SAMPLERS in ext folder
dir_name = os.path.dirname(os.path.abspath(__file__))
ext_folder = os.path.join(dir_name, "ext")

# Add the base directory (py/defs) to sys.path to ensure correct module imports
sys.path.append(os.path.join(dir_name, 'defs'))

parent_folder = os.path.basename(os.path.dirname(os.path.dirname(dir_name)))

for module_path in glob.glob(os.path.join(ext_folder, "*.py")):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    package_name = f"custom_nodes.{parent_folder}.py.defs.ext.{module_name}"

    try:
        # Import the module dynamically
        module = importlib.import_module(package_name)
        
        if hasattr(module, "CAPTURE_FIELD_LIST"):
            CAPTURE_FIELD_LIST.update(getattr(module, "CAPTURE_FIELD_LIST", {}))
        
        if hasattr(module, "SAMPLERS"):
            SAMPLERS.update(getattr(module, "SAMPLERS", {}))
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")

# Clean up sys.path after import
if os.path.join(dir_name, 'defs') in sys.path:
    sys.path.remove(os.path.join(dir_name, 'defs'))
