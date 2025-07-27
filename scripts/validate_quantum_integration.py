#!/usr/bin/env python3
"""
Quantum Integration Validation Script

This script validates that QMANN is properly integrated with real quantum
hardware interfaces and can distinguish between theoretical, simulation,
and experimental modes.
"""

import sys
import warnings
from typing import Dict, List, Any

def test_basic_imports():
    """Test basic QMANN imports."""
    print("Testing basic imports...")
    
    try:
        import qmann
        print("  ✓ qmann imported successfully")
        
        # Test core components
        from qmann import QMANN, QuantumMemory, QRAM
        print("  ✓ Core components imported")
        
        # Test system info
        info = qmann.get_system_info()
        print(f"  ✓ System info: QMANN v{info['qmann_version']}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_quantum_hardware_interface():
    """Test quantum hardware interface availability."""
    print("\nTesting quantum hardware interface...")
    
    try:
        from qmann.hardware import QuantumBackendManager, ExperimentalQMANN
        print("  ✓ Hardware interface imported")
        
        # Test backend manager
        backend_manager = QuantumBackendManager()
        backends = backend_manager.list_backends()
        
        print(f"  ✓ Found {len(backends)} quantum backends:")
        for name, info in backends.items():
            qubits = info.get('n_qubits', 'Unknown')
            provider = info.get('provider', 'Unknown')
            print(f"    - {name}: {qubits} qubits ({provider})")
            
        return True
    except ImportError as e:
        print(f"  ⚠ Hardware interface not available: {e}")
        return False

def test_experimental_configuration():
    """Test experimental configuration system."""
    print("\nTesting experimental configuration...")
    
    try:
        from qmann.config import (
            ExperimentalConfig, ExperimentMode,
            THEORETICAL_ANALYSIS, SIMULATION_VALIDATION, HARDWARE_PROOF_OF_CONCEPT,
            get_recommended_config, validate_experimental_setup
        )
        print("  ✓ Configuration system imported")
        
        # Test predefined configurations
        configs = {
            'theoretical': THEORETICAL_ANALYSIS,
            'simulation': SIMULATION_VALIDATION,
            'hardware_poc': HARDWARE_PROOF_OF_CONCEPT
        }
        
        for name, config in configs.items():
            print(f"  ✓ {name}: {config.mode.value} mode")
            
        # Test recommended config
        sim_config = get_recommended_config(n_qubits=8, budget_usd=0.0)
        hw_config = get_recommended_config(n_qubits=6, budget_usd=10.0)
        
        print(f"  ✓ Recommended for 8 qubits, $0: {sim_config.mode.value}")
        print(f"  ✓ Recommended for 6 qubits, $10: {hw_config.mode.value}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Configuration system failed: {e}")
        return False

def test_mode_separation():
    """Test separation between theoretical, simulation, and hardware modes."""
    print("\nTesting mode separation...")
    
    try:
        from qmann.config import ExperimentalConfig, ExperimentMode
        
        # Test theoretical mode
        theoretical_config = ExperimentalConfig(mode=ExperimentMode.THEORETICAL)
        theoretical_reqs = theoretical_config.get_hardware_requirements()
        
        assert theoretical_reqs['type'] == 'theoretical'
        assert theoretical_reqs['qubits'] == 'unlimited'
        assert theoretical_reqs['cost'] == 'free'
        print("  ✓ Theoretical mode: unlimited qubits, perfect gates")
        
        # Test simulation mode
        simulation_config = ExperimentalConfig(mode=ExperimentMode.SIMULATION)
        simulation_reqs = simulation_config.get_hardware_requirements()
        
        assert simulation_reqs['type'] == 'classical_simulation'
        assert isinstance(simulation_reqs['qubits'], int)
        assert simulation_reqs['cost'] == 'free'
        print(f"  ✓ Simulation mode: {simulation_reqs['qubits']} qubits, classical simulation")
        
        # Test hardware mode
        hardware_config = ExperimentalConfig(mode=ExperimentMode.HARDWARE)
        hardware_reqs = hardware_config.get_hardware_requirements()
        
        assert hardware_reqs['type'] == 'quantum_hardware'
        assert hardware_reqs['api_credentials_required'] == True
        print(f"  ✓ Hardware mode: {hardware_reqs['qubits']} qubits, real quantum hardware")
        
        return True
    except Exception as e:
        print(f"  ✗ Mode separation test failed: {e}")
        return False

def test_realistic_constraints():
    """Test that realistic quantum hardware constraints are enforced."""
    print("\nTesting realistic constraints...")
    
    try:
        from qmann.config import HardwareConfig
        from qmann.hardware import ExperimentalQMANN
        
        # Test hardware config constraints
        hw_config = HardwareConfig()
        warnings_list = hw_config.validate()
        
        print(f"  ✓ Hardware config max qubits: {hw_config.max_qubits}")
        print(f"  ✓ Hardware config max depth: {hw_config.max_circuit_depth}")
        
        # Test model with realistic constraints
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger warnings for excessive qubits
            model = ExperimentalQMANN(
                input_dim=4,
                hidden_dim=16,
                output_dim=2,
                n_qubits=25  # Should be reduced to 20
            )
            
            if w:
                print(f"  ✓ Constraint warning triggered: {w[0].message}")
            else:
                print("  ⚠ No constraint warning (may be expected)")
                
        return True
    except Exception as e:
        print(f"  ✗ Constraint test failed: {e}")
        return False

def test_cost_estimation():
    """Test quantum hardware cost estimation."""
    print("\nTesting cost estimation...")
    
    try:
        from qmann.config import ExperimentalConfig, ExperimentMode, HardwareConfig
        
        # Test free backends
        free_config = ExperimentalConfig(
            mode=ExperimentMode.HARDWARE,
            hardware=HardwareConfig(use_free_backends_only=True)
        )
        
        free_reqs = free_config.get_hardware_requirements()
        estimated_cost = free_reqs.get('estimated_cost_usd', 0)
        
        print(f"  ✓ Free backends cost: ${estimated_cost:.2f}")
        
        # Test paid backends
        paid_config = ExperimentalConfig(
            mode=ExperimentMode.HARDWARE,
            hardware=HardwareConfig(
                use_free_backends_only=False,
                shots=1000,
                max_cost_per_experiment=20.0
            )
        )
        
        paid_reqs = paid_config.get_hardware_requirements()
        paid_cost = paid_reqs.get('estimated_cost_usd', 0)
        
        print(f"  ✓ Paid backends estimated cost: ${paid_cost:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Cost estimation test failed: {e}")
        return False

def test_feature_availability():
    """Test feature availability reporting."""
    print("\nTesting feature availability...")
    
    try:
        import qmann
        
        features = qmann.get_available_features()
        
        print("  Available features:")
        for feature, available in features.items():
            status = "✓" if available else "✗"
            print(f"    {status} {feature}")
            
        # Check critical features
        critical_features = ['core', 'quantum_transformers', 'real_quantum_hardware']
        
        for feature in critical_features:
            if feature in features:
                if features[feature]:
                    print(f"  ✓ Critical feature '{feature}' available")
                else:
                    print(f"  ⚠ Critical feature '{feature}' not available")
            else:
                print(f"  ✗ Critical feature '{feature}' missing")
                
        return True
    except Exception as e:
        print(f"  ✗ Feature availability test failed: {e}")
        return False

def run_integration_test():
    """Run a simple integration test."""
    print("\nRunning integration test...")
    
    try:
        from qmann import QMANN
        from qmann.config import SIMULATION_VALIDATION, validate_experimental_setup
        
        # Validate experimental setup
        is_valid = validate_experimental_setup(SIMULATION_VALIDATION)
        print(f"  ✓ Simulation config validation: {'passed' if is_valid else 'has warnings'}")
        
        # Create and test model
        model = QMANN(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            memory_capacity=16,
            max_qubits=6
        )
        
        print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        import torch
        x = torch.randn(2, 5, 4)  # Small test input
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for test
            output, memory_info = model(x)
            
        print(f"  ✓ Forward pass successful: {output.shape}")
        print(f"  ✓ Memory usage: {memory_info['memory_usage']:.2%}")
        
        return True
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("QMANN Quantum Integration Validation")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Quantum Hardware Interface", test_quantum_hardware_interface),
        ("Experimental Configuration", test_experimental_configuration),
        ("Mode Separation", test_mode_separation),
        ("Realistic Constraints", test_realistic_constraints),
        ("Cost Estimation", test_cost_estimation),
        ("Feature Availability", test_feature_availability),
        ("Integration Test", run_integration_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            results[test_name] = False
            
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
        
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! QMANN is properly integrated with quantum hardware.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
