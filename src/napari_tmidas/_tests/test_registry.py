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

        assert len(results) == 10
        assert len(BatchProcessingRegistry.list_functions()) == 10
