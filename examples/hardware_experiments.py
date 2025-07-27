#!/usr/bin/env python3
"""
Real Quantum Hardware Experiments with QMANN

This script demonstrates how to run QMANN experiments on real quantum hardware
including IBM Quantum, Google Quantum AI, and IonQ devices.

IMPORTANT: This requires actual quantum hardware access and API credentials.
For simulation-only experiments, see basic_qmann_example.py
"""

import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# QMANN imports
try:
    from qmann.hardware import (
        QuantumBackendManager,
        ExperimentalQMANN,
        HardwareAwareQRAM,
        NISQOptimizedLayers
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    warnings.warn("Quantum hardware interface not available. Install with: pip install qmann[hardware]")

def setup_quantum_backends() -> QuantumBackendManager:
    """
    Setup quantum backends with proper credentials.
    
    Returns:
        Configured QuantumBackendManager
    """
    print("Setting up quantum backends...")
    
    # Initialize backend manager
    backend_manager = QuantumBackendManager()
    
    # Check available backends
    available_backends = backend_manager.list_backends()
    
    print("Available quantum backends:")
    for name, info in available_backends.items():
        status = "✓" if info.get('is_simulator', True) else "⚠ (Real Hardware)"
        print(f"  - {name}: {info.get('n_qubits', 'Unknown')} qubits {status}")
        
    return backend_manager

def create_test_dataset(n_samples: int = 50, seq_len: int = 8, 
                       input_dim: int = 4) -> tuple:
    """
    Create small test dataset suitable for quantum hardware experiments.
    
    Args:
        n_samples: Number of samples (keep small for hardware costs)
        seq_len: Sequence length (limited by quantum memory)
        input_dim: Input dimension (limited by qubits)
        
    Returns:
        Tuple of (X, y) tensors
    """
    print(f"Creating test dataset: {n_samples} samples, {seq_len} sequence length, {input_dim} features")
    
    # Generate simple pattern recognition task
    X = torch.randn(n_samples, seq_len, input_dim)
    
    # Create labels based on simple pattern (sum of first two features)
    pattern_scores = torch.sum(X[:, :, :2], dim=2)  # [n_samples, seq_len]
    y = (pattern_scores > 0).long()  # Binary classification
    
    return X, y

def run_simulator_validation(backend_manager: QuantumBackendManager) -> Dict:
    """
    Run validation experiments on quantum simulators first.
    
    Args:
        backend_manager: Quantum backend manager
        
    Returns:
        Validation results
    """
    print("\n" + "="*50)
    print("PHASE 1: SIMULATOR VALIDATION")
    print("="*50)
    
    # Create small test dataset
    X_test, y_test = create_test_dataset(n_samples=20, seq_len=5, input_dim=4)
    
    # Initialize experimental QMANN
    model = ExperimentalQMANN(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        n_qubits=6,  # Small for initial testing
        backend_manager=backend_manager
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test on different simulator backends
    simulator_results = {}
    
    for backend_name in ['ibm_simulator', 'google_simulator']:
        if backend_name in backend_manager.backends:
            print(f"\nTesting on {backend_name}...")
            
            try:
                # Run forward pass
                output, exp_info = model.experimental_forward(
                    X_test[:5],  # Small batch
                    backend_name=backend_name,
                    log_experiment=True
                )
                
                simulator_results[backend_name] = {
                    'success': exp_info['quantum_success'],
                    'output_shape': output.shape,
                    'backend_info': exp_info.get('backend_info', {}),
                    'hardware_stats': exp_info.get('hardware_stats', {})
                }
                
                print(f"  ✓ Success: {exp_info['quantum_success']}")
                print(f"  ✓ Output shape: {output.shape}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                simulator_results[backend_name] = {'error': str(e)}
                
    return simulator_results

def run_hardware_experiments(backend_manager: QuantumBackendManager,
                            use_real_hardware: bool = False) -> Dict:
    """
    Run experiments on real quantum hardware (if available and enabled).
    
    Args:
        backend_manager: Quantum backend manager
        use_real_hardware: Whether to use real hardware (costs money!)
        
    Returns:
        Hardware experiment results
    """
    print("\n" + "="*50)
    print("PHASE 2: HARDWARE EXPERIMENTS")
    print("="*50)
    
    if not use_real_hardware:
        print("⚠ Real hardware experiments disabled (use_real_hardware=False)")
        print("  To enable: Set use_real_hardware=True and ensure proper API credentials")
        return {"message": "Hardware experiments skipped"}
        
    # Create minimal test dataset for hardware (costs!)
    X_test, y_test = create_test_dataset(n_samples=5, seq_len=3, input_dim=3)
    
    # Initialize hardware-optimized model
    model = ExperimentalQMANN(
        input_dim=3,
        hidden_dim=8,
        output_dim=2,
        n_qubits=4,  # Very small for real hardware
        backend_manager=backend_manager
    )
    
    print("⚠ WARNING: Real hardware experiments will incur costs!")
    print("  - IBM Quantum: ~$1.60 per second of QPU time")
    print("  - IonQ: ~$0.01 per shot")
    print("  - Google: Varies by processor")
    
    # Run hardware benchmark
    hardware_results = model.run_hardware_benchmark(
        test_data=X_test,
        backends=['ibm_real', 'ionq_real'] if use_real_hardware else []
    )
    
    return hardware_results

def analyze_quantum_advantage(results: Dict) -> Dict:
    """
    Analyze potential quantum advantage from experimental results.
    
    Args:
        results: Combined experimental results
        
    Returns:
        Quantum advantage analysis
    """
    print("\n" + "="*50)
    print("PHASE 3: QUANTUM ADVANTAGE ANALYSIS")
    print("="*50)
    
    analysis = {
        'simulator_performance': {},
        'hardware_performance': {},
        'quantum_advantage_indicators': []
    }
    
    # Analyze simulator results
    if 'simulator' in results:
        sim_results = results['simulator']
        
        success_rates = {}
        for backend, result in sim_results.items():
            if 'success' in result:
                success_rates[backend] = result['success']
                
        analysis['simulator_performance'] = {
            'average_success_rate': np.mean(list(success_rates.values())) if success_rates else 0,
            'backend_success_rates': success_rates
        }
        
        print(f"Simulator average success rate: {analysis['simulator_performance']['average_success_rate']:.2%}")
        
    # Analyze hardware results
    if 'hardware' in results and results['hardware'].get('message') != "Hardware experiments skipped":
        hw_results = results['hardware']
        
        for backend, result in hw_results.items():
            if 'success_rate' in result:
                analysis['hardware_performance'][backend] = result['success_rate']
                
        print(f"Hardware performance: {analysis['hardware_performance']}")
        
    # Quantum advantage indicators
    indicators = []
    
    # 1. Circuit depth advantage
    indicators.append({
        'metric': 'Circuit Depth',
        'description': 'Quantum circuits can represent exponentially complex operations',
        'evidence': 'Theoretical - 2^n state space with n qubits'
    })
    
    # 2. Memory capacity advantage
    indicators.append({
        'metric': 'Memory Capacity',
        'description': 'Quantum superposition enables exponential memory scaling',
        'evidence': 'Amplitude encoding allows 2^n classical values in n qubits'
    })
    
    # 3. Parallel processing
    indicators.append({
        'metric': 'Parallel Processing',
        'description': 'Quantum operations act on all amplitudes simultaneously',
        'evidence': 'Single quantum gate affects entire superposition state'
    })
    
    analysis['quantum_advantage_indicators'] = indicators
    
    print("\nQuantum Advantage Indicators:")
    for i, indicator in enumerate(indicators, 1):
        print(f"  {i}. {indicator['metric']}: {indicator['description']}")
        
    return analysis

def create_experimental_report(results: Dict, analysis: Dict) -> str:
    """
    Create comprehensive experimental report.
    
    Args:
        results: Experimental results
        analysis: Quantum advantage analysis
        
    Returns:
        Report string
    """
    report = []
    report.append("QMANN QUANTUM HARDWARE EXPERIMENTAL REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    
    if 'simulator' in results:
        sim_success = analysis['simulator_performance']['average_success_rate']
        report.append(f"• Simulator Success Rate: {sim_success:.2%}")
        
    if 'hardware' in results and results['hardware'].get('message') != "Hardware experiments skipped":
        report.append("• Real Hardware: Experiments completed")
    else:
        report.append("• Real Hardware: Experiments skipped (simulation only)")
        
    report.append("")
    
    # Technical Details
    report.append("TECHNICAL DETAILS")
    report.append("-" * 20)
    
    # Simulator results
    if 'simulator' in results:
        report.append("Simulator Results:")
        for backend, result in results['simulator'].items():
            if 'success' in result:
                report.append(f"  - {backend}: {'✓' if result['success'] else '✗'}")
                
    report.append("")
    
    # Quantum Advantage Analysis
    report.append("QUANTUM ADVANTAGE ANALYSIS")
    report.append("-" * 30)
    
    for indicator in analysis['quantum_advantage_indicators']:
        report.append(f"• {indicator['metric']}")
        report.append(f"  {indicator['description']}")
        report.append(f"  Evidence: {indicator['evidence']}")
        report.append("")
        
    # Limitations and Future Work
    report.append("LIMITATIONS AND FUTURE WORK")
    report.append("-" * 35)
    report.append("• Current experiments limited to small qubit counts (4-8 qubits)")
    report.append("• NISQ devices have high error rates (~0.1-1%)")
    report.append("• Classical simulation still required for validation")
    report.append("• Real quantum advantage requires fault-tolerant quantum computers")
    report.append("")
    report.append("Future Work:")
    report.append("• Scale to larger quantum devices (IBM Starling 2029: 200 logical qubits)")
    report.append("• Implement advanced error correction (surface codes)")
    report.append("• Develop quantum-specific machine learning algorithms")
    report.append("• Benchmark against classical supercomputers")
    
    return "\n".join(report)

def main():
    """Main experimental workflow."""
    print("QMANN Quantum Hardware Experiments")
    print("=" * 40)
    
    if not HARDWARE_AVAILABLE:
        print("❌ Quantum hardware interface not available!")
        print("Install with: pip install qmann[hardware]")
        return
        
    # Configuration
    USE_REAL_HARDWARE = False  # Set to True for real hardware experiments (costs money!)
    
    print(f"Configuration:")
    print(f"  - Real Hardware: {'Enabled' if USE_REAL_HARDWARE else 'Disabled (simulation only)'}")
    print(f"  - Safety: {'⚠ COSTS MONEY' if USE_REAL_HARDWARE else '✓ Free simulation'}")
    print()
    
    try:
        # Phase 1: Setup
        backend_manager = setup_quantum_backends()
        
        # Phase 2: Simulator validation
        simulator_results = run_simulator_validation(backend_manager)
        
        # Phase 3: Hardware experiments (if enabled)
        hardware_results = run_hardware_experiments(backend_manager, USE_REAL_HARDWARE)
        
        # Phase 4: Analysis
        combined_results = {
            'simulator': simulator_results,
            'hardware': hardware_results
        }
        
        analysis = analyze_quantum_advantage(combined_results)
        
        # Phase 5: Report
        report = create_experimental_report(combined_results, analysis)
        
        print("\n" + "="*50)
        print("EXPERIMENTAL REPORT")
        print("="*50)
        print(report)
        
        # Save report
        with open('qmann_hardware_experiment_report.txt', 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: qmann_hardware_experiment_report.txt")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
