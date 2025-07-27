"""
QMANN Test Suite

This package contains comprehensive tests for the QMANN system.
Tests are designed to work across different environments:
- Theoretical mode (pure mathematics)
- Simulation mode (classical simulation with Qiskit/PennyLane)
- Hardware mode (real quantum devices when available)

Test Structure:
- test_core.py: Core QRAM and QuantumMemory tests
- test_models.py: QMANN model tests
- test_training.py: Training and optimization tests
- test_hardware.py: Hardware interface and backend tests
- test_integration.py: End-to-end integration tests

Usage:
    # Run all tests
    python -m unittest discover tests/

    # Run specific test module
    python -m unittest tests.test_core

    # Run with verbose output
    python -m unittest discover tests/ -v

    # Run specific test class
    python -m unittest tests.test_core.TestQRAM

    # Run specific test method
    python -m unittest tests.test_core.TestQRAM.test_qram_initialization
"""

import unittest
import warnings
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure warnings for tests
warnings.filterwarnings("ignore", category=UserWarning, module="qmann")
warnings.filterwarnings("ignore", category=FutureWarning)


def detect_test_environment():
    """Detect the testing environment and available features."""
    environment = {
        'has_qiskit': False,
        'has_pennylane': False,
        'has_cuda': False,
        'has_hardware_interface': False,
        'mode': 'theoretical'
    }
    
    # Check for Qiskit
    try:
        import qiskit
        environment['has_qiskit'] = True
        environment['mode'] = 'simulation'
    except ImportError:
        pass
    
    # Check for PennyLane
    try:
        import pennylane
        environment['has_pennylane'] = True
        if environment['mode'] == 'theoretical':
            environment['mode'] = 'simulation'
    except ImportError:
        pass
    
    # Check for CUDA
    try:
        import torch
        environment['has_cuda'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check for hardware interface
    try:
        from qmann.hardware import QuantumBackendManager
        environment['has_hardware_interface'] = True
    except ImportError:
        pass
    
    return environment


def create_test_suite():
    """Create a comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load all test modules
    test_modules = [
        'tests.test_core',
        'tests.test_models',
        'tests.test_training',
        'tests.test_hardware',
        'tests.test_integration'
    ]
    
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
        except ImportError as e:
            print(f"Warning: Could not load {module}: {e}")
    
    return suite


def run_tests(verbosity=2, pattern=None):
    """Run the test suite with specified verbosity."""
    # Detect environment
    env = detect_test_environment()
    
    print("QMANN Test Suite")
    print("=" * 50)
    print(f"Environment: {env['mode'].upper()}")
    print(f"Qiskit: {'✓' if env['has_qiskit'] else '✗'}")
    print(f"PennyLane: {'✓' if env['has_pennylane'] else '✗'}")
    print(f"CUDA: {'✓' if env['has_cuda'] else '✗'}")
    print(f"Hardware Interface: {'✓' if env['has_hardware_interface'] else '✗'}")
    print("=" * 50)
    
    # Create and run test suite
    if pattern:
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern=pattern)
    else:
        suite = create_test_suite()
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All tests passed!")
    elif success_rate >= 90:
        print("✅ Most tests passed - system is functional")
    elif success_rate >= 70:
        print("⚠️ Some tests failed - check compatibility")
    else:
        print("❌ Many tests failed - check installation")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Allow running tests directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Run QMANN test suite")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("-p", "--pattern", type=str, 
                       help="Test file pattern (e.g., 'test_core*')")
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    success = run_tests(verbosity=verbosity, pattern=args.pattern)
    
    sys.exit(0 if success else 1)
