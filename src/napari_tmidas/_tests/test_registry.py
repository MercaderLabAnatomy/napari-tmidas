# src/napari_tmidas/_tests/test_registry.py
from napari_tmidas._registry import BatchProcessingRegistry


class TestBatchProcessingRegistry:
    def setup_method(self):
        """Clear registry before each test"""
        BatchProcessingRegistry._processing_functions.clear()

    def test_register_function(self):
        """Test registering a processing function"""

        @BatchProcessingRegistry.register(
            name="Test Function",
            suffix="_test",
            description="Test description",
            parameters={"param1": {"type": int, "default": 5}},
        )
        def test_func(image, param1=5):
            return image + param1

        assert "Test Function" in BatchProcessingRegistry.list_functions()
        info = BatchProcessingRegistry.get_function_info("Test Function")
        assert info["suffix"] == "_test"
        assert info["description"] == "Test description"
        assert info["func"] == test_func

    def test_list_functions(self):
        """Test listing registered functions"""

        @BatchProcessingRegistry.register(name="Func1")
        def func1(image):
            return image

        @BatchProcessingRegistry.register(name="Func2")
        def func2(image):
            return image

        functions = BatchProcessingRegistry.list_functions()
        assert len(functions) == 2
        assert "Func1" in functions
        assert "Func2" in functions

    def test_thread_safety(self):
        """Test thread-safe registration"""
        import threading

        results = []

        def register_func(i):
            @BatchProcessingRegistry.register(name=f"ThreadFunc{i}")
            def func(image):
                return image

            results.append(i)

        threads = [
            threading.Thread(target=register_func, args=(i,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_get_function_info_nonexistent(self):
        """Test getting info for non-existent function"""
        info = BatchProcessingRegistry.get_function_info("NonExistent")
        assert info is None

    def test_register_with_none_parameters(self):
        """Test registering function with None parameters (should convert to empty dict)"""

        @BatchProcessingRegistry.register(
            name="None Params Function",
            suffix="_none",
            description="Function with None parameters",
        )
        def none_params_func(image):
            return image

        info = BatchProcessingRegistry.get_function_info(
            "None Params Function"
        )
        assert info["parameters"] == {}
        assert info["suffix"] == "_none"
        assert info["description"] == "Function with None parameters"

    def test_register_minimal(self):
        """Test registering function with minimal parameters"""

        @BatchProcessingRegistry.register(name="Minimal Function")
        def minimal_func(image):
            return image

        info = BatchProcessingRegistry.get_function_info("Minimal Function")
        assert info is not None
        assert callable(info["func"])

    def test_register_with_complex_parameters(self):
        """Test registering function with complex parameter metadata"""

        @BatchProcessingRegistry.register(
            name="Complex Params Function",
            suffix="_complex",
            description="Function with complex parameters",
            parameters={
                "param1": {
                    "type": int,
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "description": "Parameter description",
                },
                "param2": {
                    "type": str,
                    "default": "default_value",
                    "description": "String parameter",
                },
            },
        )
        def complex_func(image, param1=5, param2="default"):
            return image

        info = BatchProcessingRegistry.get_function_info(
            "Complex Params Function"
        )
        assert info["parameters"]["param1"]["type"] is int
        assert info["parameters"]["param1"]["default"] == 5
        assert info["parameters"]["param1"]["min"] == 1
        assert info["parameters"]["param1"]["max"] == 10
        assert (
            info["parameters"]["param1"]["description"]
            == "Parameter description"
        )
