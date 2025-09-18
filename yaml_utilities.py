#!/usr/bin/env python3
"""
YAML Utilities with Robust Import Handling

This module provides a robust way to handle YAML parsing that works
with different YAML module installations.
"""

def safe_yaml_load(stream):
    """
    Safely load YAML content with fallback handling for different YAML modules.
    
    Args:
        stream: File-like object or string containing YAML content
        
    Returns:
        Parsed YAML content as Python object
        
    Raises:
        ImportError: If no suitable YAML module is available
        yaml.YAMLError: If YAML parsing fails
    """
    # Try different YAML import strategies
    yaml_module = None
    safe_load_func = None
    yaml_error_class = None
    
    # Strategy 1: Try importing yaml and check for safe_load
    try:
        import yaml
        if hasattr(yaml, 'safe_load'):
            yaml_module = yaml
            safe_load_func = yaml.safe_load
            yaml_error_class = getattr(yaml, 'YAMLError', Exception)
        else:
            # Strategy 2: Try importing specific functions from yaml package
            try:
                from yaml import safe_load, YAMLError
                safe_load_func = safe_load
                yaml_error_class = YAMLError
            except ImportError:
                # Strategy 3: Try yaml.load with FullLoader
                if hasattr(yaml, 'load') and hasattr(yaml, 'FullLoader'):
                    def safe_load_wrapper(stream):
                        return yaml.load(stream, Loader=yaml.FullLoader)
                    safe_load_func = safe_load_wrapper
                    yaml_error_class = getattr(yaml, 'YAMLError', Exception)
    except ImportError:
        pass
    
    # If we still don't have a working YAML parser, try PyYAML specifically
    if safe_load_func is None:
        try:
            import yaml as pyyaml
            if hasattr(pyyaml, 'safe_load'):
                yaml_module = pyyaml
                safe_load_func = pyyaml.safe_load
                yaml_error_class = getattr(pyyaml, 'YAMLError', Exception)
        except ImportError:
            pass
    
    # If we still don't have a working YAML parser, raise an error
    if safe_load_func is None:
        raise ImportError(
            "No suitable YAML module found. Please install PyYAML: pip install PyYAML"
        )
    
    # Parse the YAML content
    try:
        return safe_load_func(stream)
    except Exception as e:
        if yaml_error_class and isinstance(e, yaml_error_class):
            raise e
        else:
            # Re-raise as YAMLError if we can determine the error class
            if yaml_error_class:
                raise yaml_error_class(str(e))
            else:
                raise Exception(f"YAML parsing failed: {e}")


def safe_yaml_dump(data, stream=None, **kwargs):
    """
    Safely dump Python object to YAML with fallback handling.
    
    Args:
        data: Python object to serialize
        stream: File-like object to write to (optional)
        **kwargs: Additional arguments for YAML dumper
        
    Returns:
        YAML string if stream is None, otherwise None
    """
    # Try different YAML import strategies
    yaml_module = None
    dump_func = None
    
    # Strategy 1: Try importing yaml and check for dump
    try:
        import yaml
        if hasattr(yaml, 'dump'):
            yaml_module = yaml
            dump_func = yaml.dump
        else:
            # Strategy 2: Try importing specific functions from yaml package
            try:
                from yaml import dump
                dump_func = dump
            except ImportError:
                pass
    except ImportError:
        pass
    
    # If we still don't have a working YAML dumper, try PyYAML specifically
    if dump_func is None:
        try:
            import yaml as pyyaml
            if hasattr(pyyaml, 'dump'):
                yaml_module = pyyaml
                dump_func = pyyaml.dump
        except ImportError:
            pass
    
    # If we still don't have a working YAML dumper, raise an error
    if dump_func is None:
        raise ImportError(
            "No suitable YAML module found. Please install PyYAML: pip install PyYAML"
        )
    
    # Dump the data
    return dump_func(data, stream, **kwargs)


# Test the YAML functionality
if __name__ == "__main__":
    print("Testing YAML utilities...")
    
    test_data = {
        'model': {
            'name': 'test_model',
            'input_shape': [1, 3, 32, 32],
            'num_classes': 10
        },
        'layers': [
            {'type': 'Linear', 'params': {'in_features': 100, 'out_features': 10}}
        ]
    }
    
    try:
        # Test YAML dumping
        yaml_string = safe_yaml_dump(test_data)
        print("✓ YAML dumping successful")
        
        # Test YAML loading
        from io import StringIO
        loaded_data = safe_yaml_load(StringIO(yaml_string))
        print("✓ YAML loading successful")
        
        # Verify data integrity
        if loaded_data == test_data:
            print("✓ Data integrity verified")
        else:
            print("✗ Data integrity check failed")
            
    except Exception as e:
        print(f"✗ YAML test failed: {e}")
