from typing import Optional
import tempfile
import shutil
import contextlib

@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        
def update_nested_dict(nested_dict, update_dict):
    for k, v in update_dict.items():
        if isinstance(nested_dict, dict) and k in nested_dict.keys(): 
           nested_dict[k] = v
        elif isinstance(nested_dict, dict):
            for key in nested_dict.keys(): 
              update_nested_dict(nested_dict[key], update_dict)