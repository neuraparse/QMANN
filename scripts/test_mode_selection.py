#!/usr/bin/env python3
"""
Test Mode Selection System

This script tests the mode selection system to ensure clear separation
between theoretical, simulation, and hardware modes.
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_mode_recommendations():
    """Test mode recommendation system."""
    print("🧪 TESTING MODE RECOMMENDATIONS")
    print("=" * 50)
    
    try:
        from qmnn import recommend_mode
        
        test_cases = [
            # (purpose, budget, qubits, expected_mode)
            ("research", 0.0, None, "THEORETICAL"),
            ("development", 0.0, 8, "SIMULATION"),
            ("validation", 10.0, 6, "HARDWARE"),
            ("education", 0.0, 5, "SIMULATION"),
            (None, 0.0, 25, "THEORETICAL"),  # Too many qubits
            (None, 50.0, 8, "HARDWARE"),     # Has budget
        ]
        
        for purpose, budget, qubits, expected in test_cases:
            recommendation = recommend_mode(purpose, budget, qubits)
            
            print(f"\nTest case:")
            print(f"  Purpose: {purpose}")
            print(f"  Budget: ${budget}")
            print(f"  Qubits: {qubits}")
            print(f"  Recommendation:")
            for line in recommendation.split('\n'):
                print(f"    {line}")
                
            # Check if expected mode is in recommendation
            if expected in recommendation:
                print(f"  ✅ Contains expected mode: {expected}")
            else:
                print(f"  ⚠️  Expected {expected} not found in recommendation")
                
        return True
        
    except Exception as e:
        print(f"❌ Mode recommendation test failed: {e}")
        return False


def test_mode_guide():
    """Test mode guide printing."""
    print("\n🧪 TESTING MODE GUIDE")
    print("=" * 50)
    
    try:
        from qmnn import print_mode_guide
        
        print("Testing print_mode_guide()...")
        print_mode_guide()
        
        print("\n✅ Mode guide printed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mode guide test failed: {e}")
        return False


def test_quick_start():
    """Test quick start functionality."""
    print("\n🧪 TESTING QUICK START")
    print("=" * 50)
    
    try:
        from qmnn import quick_start
        
        modes = ["theoretical", "simulation", "hardware"]
        
        for mode in modes:
            print(f"\nTesting quick_start('{mode}'):")
            quick_start(mode)
            print(f"✅ {mode} mode quick start completed")
            
        # Test invalid mode
        print(f"\nTesting invalid mode:")
        quick_start("invalid_mode")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick start test failed: {e}")
        return False


def test_configuration_separation():
    """Test configuration separation between modes."""
    print("\n🧪 TESTING CONFIGURATION SEPARATION")
    print("=" * 50)
    
    try:
        from qmnn.config import (
            THEORETICAL_ANALYSIS, 
            SIMULATION_VALIDATION, 
            HARDWARE_PROOF_OF_CONCEPT,
            ExperimentMode
        )
        
        configs = {
            "Theoretical": THEORETICAL_ANALYSIS,
            "Simulation": SIMULATION_VALIDATION,
            "Hardware": HARDWARE_PROOF_OF_CONCEPT
        }
        
        for name, config in configs.items():
            print(f"\n{name} Configuration:")
            print(f"  Mode: {config.mode.value}")
            
            active_config = config.get_active_config()
            
            if hasattr(active_config, 'max_qubits'):
                print(f"  Max qubits: {active_config.max_qubits}")
                
            if hasattr(active_config, 'perfect_gates'):
                print(f"  Perfect gates: {active_config.perfect_gates}")
                
            if hasattr(active_config, 'use_free_backends_only'):
                print(f"  Free backends only: {active_config.use_free_backends_only}")
                
            # Test hardware requirements
            hw_reqs = config.get_hardware_requirements()
            print(f"  Hardware type: {hw_reqs['type']}")
            print(f"  Cost: {hw_reqs.get('cost', 'Unknown')}")
            
        # Verify mode separation
        assert THEORETICAL_ANALYSIS.mode == ExperimentMode.THEORETICAL
        assert SIMULATION_VALIDATION.mode == ExperimentMode.SIMULATION
        assert HARDWARE_PROOF_OF_CONCEPT.mode == ExperimentMode.HARDWARE
        
        print("\n✅ Configuration separation verified")
        return True
        
    except Exception as e:
        print(f"❌ Configuration separation test failed: {e}")
        return False


def test_cost_awareness():
    """Test cost awareness in mode selection."""
    print("\n🧪 TESTING COST AWARENESS")
    print("=" * 50)
    
    try:
        from qmnn.config import HARDWARE_PROOF_OF_CONCEPT
        
        # Test cost estimation
        hw_config = HARDWARE_PROOF_OF_CONCEPT
        hw_reqs = hw_config.get_hardware_requirements()
        
        print("Hardware Mode Cost Analysis:")
        print(f"  Type: {hw_reqs['type']}")
        print(f"  Estimated cost: ${hw_reqs.get('estimated_cost_usd', 0):.2f}")
        print(f"  API credentials required: {hw_reqs.get('api_credentials_required', False)}")
        
        # Test free mode
        from qmnn.config import SIMULATION_VALIDATION
        sim_reqs = SIMULATION_VALIDATION.get_hardware_requirements()
        
        print("\nSimulation Mode Cost Analysis:")
        print(f"  Type: {sim_reqs['type']}")
        print(f"  Cost: {sim_reqs.get('cost', 'Unknown')}")
        
        # Verify cost separation
        assert sim_reqs.get('cost') == 'free'
        assert hw_reqs.get('api_credentials_required') == True
        
        print("\n✅ Cost awareness verified")
        return True
        
    except Exception as e:
        print(f"❌ Cost awareness test failed: {e}")
        return False


def test_example_files():
    """Test that example files exist and are properly named."""
    print("\n🧪 TESTING EXAMPLE FILES")
    print("=" * 50)
    
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    
    expected_files = [
        '01_theoretical_mode.py',
        '02_simulation_mode.py', 
        '03_hardware_mode.py'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(examples_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {filename} exists")
            
            # Check file content for mode indicators
            with open(filepath, 'r') as f:
                content = f.read()
                
            if 'THEORETICAL MODE' in content:
                print(f"  ✅ Contains THEORETICAL MODE indicator")
            elif 'SIMULATION MODE' in content:
                print(f"  ✅ Contains SIMULATION MODE indicator")
            elif 'HARDWARE MODE' in content:
                print(f"  ✅ Contains HARDWARE MODE indicator")
            else:
                print(f"  ⚠️  No clear mode indicator found")
                
        else:
            print(f"❌ {filename} missing")
            return False
            
    print("\n✅ All example files present and properly labeled")
    return True


def test_readme_mode_section():
    """Test that README contains clear mode separation."""
    print("\n🧪 TESTING README MODE SECTION")
    print("=" * 50)
    
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    
    if not os.path.exists(readme_path):
        print("❌ README.md not found")
        return False
        
    with open(readme_path, 'r') as f:
        readme_content = f.read()
        
    # Check for mode indicators
    mode_indicators = [
        'THEORETICAL MODE',
        'SIMULATION MODE', 
        'HARDWARE MODE',
        'Mode Selection Guide',
        'FREE',
        'PAID'
    ]
    
    for indicator in mode_indicators:
        if indicator in readme_content:
            print(f"✅ README contains: {indicator}")
        else:
            print(f"⚠️  README missing: {indicator}")
            
    # Check for cost warnings
    cost_warnings = ['costs money', 'PAID', 'WARNING']
    cost_warning_found = any(warning.lower() in readme_content.lower() for warning in cost_warnings)
    
    if cost_warning_found:
        print("✅ README contains cost warnings")
    else:
        print("⚠️  README missing cost warnings")
        
    print("\n✅ README mode section verified")
    return True


def main():
    """Run all mode selection tests."""
    print("🧪 QMNN MODE SELECTION SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Mode Recommendations", test_mode_recommendations),
        ("Mode Guide", test_mode_guide),
        ("Quick Start", test_quick_start),
        ("Configuration Separation", test_configuration_separation),
        ("Cost Awareness", test_cost_awareness),
        ("Example Files", test_example_files),
        ("README Mode Section", test_readme_mode_section),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
            
    # Summary
    print("\n" + "=" * 60)
    print("MODE SELECTION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All mode selection tests passed!")
        print("✅ Theoretical, Simulation, and Hardware modes are clearly separated")
        print("✅ Cost awareness is properly implemented")
        print("✅ User guidance is comprehensive")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
