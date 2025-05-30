# napari_tmidas/_registry.py
"""
Registry for batch processing functions.
"""
import threading
from typing import Any, Dict, List, Optional


class BatchProcessingRegistry:
    """
    A registry to manage and track available processing functions with parameter support
    """

    _processing_functions = {}
    _lock = threading.RLock()  # Add thread lock

    @classmethod
    def register(
        cls,
        name: str,
        suffix: str = "_processed",
        description: str = "",
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Decorator to register processing functions

        Args:
            name: Name of the processing function
            suffix: Suffix to append to processed files
            description: Description of what the function does
            parameters: Dictionary of parameters with their metadata
                {
                    "param_name": {
                        "type": type,
                        "default": default_value,
                        "min": min_value,  # optional, for numeric types
                        "max": max_value,  # optional, for numeric types
                        "description": "Parameter description"
                    },
                    ...
                }
        """
        if parameters is None:
            parameters = {}

        def decorator(func):
            with cls._lock:  # Thread-safe registration
                cls._processing_functions[name] = {
                    "func": func,
                    "suffix": suffix,
                    "description": description,
                    "parameters": parameters,
                }
            return func

        return decorator

    @classmethod
    def get_function_info(cls, name: str) -> Optional[dict]:
        """Thread-safe retrieval"""
        with cls._lock:
            return cls._processing_functions.get(name)

    @classmethod
    def list_functions(cls) -> List[str]:
        """Thread-safe listing"""
        with cls._lock:
            return list(cls._processing_functions.keys())
