#!/usr/bin/env python
"""
Diagnostic script to test for potential C extension issues that might cause segfaults.
"""
import sys
import traceback


def test_imports():
    """Test importing critical modules that use C extensions."""
    modules_to_test = [
        "numpy",
        "scipy",
        "skimage",
        "torch",
        "torchvision",
        "tifffile",
        "cv2",
        "psygnal",
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            failed_imports.append(module)
        except (ModuleNotFoundError, RuntimeError, SystemError) as e:
            print(f"✗ {module} import error: {e}")
            failed_imports.append(module)

    return failed_imports


def test_numpy_operations():
    """Test basic numpy operations."""
    try:
        import numpy as np

        # Test basic array operations
        arr = np.random.rand(100, 100)
        result = np.sum(arr)
        print(f"✓ Numpy operations work: sum = {result}")

        # Test memory operations
        arr2 = np.zeros((1000, 1000), dtype=np.float64)
        arr2[:] = 1.0
        print(f"✓ Numpy memory operations work: shape = {arr2.shape}")

        return True
    except (ValueError, TypeError, RuntimeError, MemoryError) as e:
        print(f"✗ Numpy operations failed: {e}")
        traceback.print_exc()
        return False


def test_qt_operations():
    """Test Qt operations."""
    try:
        import sys

        from qtpy.QtWidgets import QApplication

        # Only test if we don't have a QApplication already
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        print("✓ Qt operations work")
        return True
    except (ImportError, RuntimeError, SystemError) as e:
        print(f"✗ Qt operations failed: {e}")
        return False


def main():
    print("Running diagnostic tests for C extension issues...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports:")
    failed_imports = test_imports()

    # Test numpy operations
    print("\n2. Testing numpy operations:")
    numpy_ok = test_numpy_operations()

    # Test Qt operations
    print("\n3. Testing Qt operations:")
    qt_ok = test_qt_operations()

    print("\n" + "=" * 50)
    print("Diagnostic Summary:")

    if failed_imports:
        print(f"Failed imports: {', '.join(failed_imports)}")

    if not numpy_ok:
        print("Numpy operations failed")

    if not qt_ok:
        print("Qt operations failed")

    if not failed_imports and numpy_ok and qt_ok:
        print("✓ All diagnostic tests passed")
        return 0
    else:
        print("✗ Some diagnostic tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
