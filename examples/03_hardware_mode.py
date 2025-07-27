#!/usr/bin/env python3
"""
QMANN Hardware Mode Example

This example demonstrates real quantum hardware experiments with QMANN,
including cost estimation, safety checks, and experimental validation.

⚛️ HARDWARE MODE:
- Purpose: Experimental validation on real quantum devices
- Resources: 4-12 qubits, real noise, actual quantum effects
- Cost: PAID (IBM: ~$0.001/shot, IonQ: ~$0.01/shot)
- Use case: Proof-of-concept, hardware benchmarking, quantum advantage validation

⚠️ WARNING: This mode costs real money! Always estimate costs first.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Optional

# QMANN imports
try:
    from qmann.hardware import ExperimentalQMANN, QuantumBackendManager
    from qmann.config import HARDWARE_PROOF_OF_CONCEPT, validate_experimental_setup
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("❌ Hardware interface not available!")
    print("Install with: pip install qmann[hardware]")


def check_hardware_prerequisites() -> bool:
    """Check if hardware prerequisites are met."""
    print("🔍 CHECKING HARDWARE PREREQUISITES")
    print("=" * 50)
    
    if not HARDWARE_AVAILABLE:
        print("❌ Hardware interface not installed")
        return False
        
    # Check API credentials
    credentials_found = False
    
    if os.getenv('IBMQ_TOKEN'):
        print("✅ IBM Quantum token found")
        credentials_found = True
    else:
        print("⚠️  IBM Quantum token not found (set IBMQ_TOKEN)")
        
    if os.getenv('GOOGLE_QUANTUM_PROJECT'):
        print("✅ Google Quantum project found")
        credentials_found = True
    else:
        print("⚠️  Google Quantum project not found (set GOOGLE_QUANTUM_PROJECT)")
        
    if os.getenv('IONQ_API_KEY'):
        print("✅ IonQ API key found")
        credentials_found = True
    else:
        print("⚠️  IonQ API key not found (set IONQ_API_KEY)")
        
    if not credentials_found:
        print("\n⚠️  No quantum hardware credentials found!")
        print("   Will use simulators only (free)")
        
    return True


def estimate_experiment_costs(n_qubits: int, n_experiments: int, shots_per_experiment: int) -> Dict:
    """Estimate costs for hardware experiments."""
    print(f"\n💰 COST ESTIMATION")
    print("=" * 30)
    
    # Import cost estimation
    sys.path.append('scripts')
    try:
        from estimate_hardware_costs import estimate_experiment_cost
        
        backends = ['ibm_brisbane', 'ionq_aria']
        total_cost = 0.0
        cost_breakdown = {}
        
        for backend in backends:
            estimate = estimate_experiment_cost(
                backend_name=backend,
                n_qubits=n_qubits,
                shots=shots_per_experiment,
                n_experiments=n_experiments
            )
            
            if 'error' not in estimate:
                cost_breakdown[backend] = estimate['total_cost']
                total_cost += estimate['total_cost']
                
                print(f"{estimate['backend']}:")
                print(f"  - Total shots: {estimate['total_shots']:,}")
                print(f"  - Free shots: {estimate['free_shots_used']:,}")
                print(f"  - Paid shots: {estimate['paid_shots']:,}")
                print(f"  - Cost: ${estimate['total_cost']:.2f}")
            else:
                print(f"{backend}: {estimate['error']}")
                
        print(f"\n💵 TOTAL ESTIMATED COST: ${total_cost:.2f}")
        
        return {
            'total_cost': total_cost,
            'breakdown': cost_breakdown,
            'n_qubits': n_qubits,
            'n_experiments': n_experiments,
            'shots_per_experiment': shots_per_experiment
        }
        
    except ImportError:
        print("⚠️  Cost estimation script not available")
        return {'total_cost': 0.0, 'breakdown': {}}


def safety_confirmation(cost_estimate: Dict) -> bool:
    """Get user confirmation for paid experiments."""
    total_cost = cost_estimate.get('total_cost', 0.0)
    
    if total_cost == 0.0:
        print("✅ FREE experiment (simulators only)")
        return True
        
    print(f"\n⚠️  COST CONFIRMATION REQUIRED")
    print("=" * 40)
    print(f"Estimated cost: ${total_cost:.2f}")
    print("This will charge your quantum cloud accounts!")
    print()
    
    # Cost warnings
    if total_cost > 100:
        print("🚨 HIGH COST WARNING (>$100)")
        print("   This is an expensive experiment!")
    elif total_cost > 10:
        print("⚠️  MODERATE COST WARNING (>$10)")
        print("   Ensure you have budget approval")
        
    response = input("Do you want to proceed with paid hardware experiments? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("✅ Proceeding with hardware experiments...")
        return True
    else:
        print("❌ Hardware experiments cancelled")
        print("   Will run simulator validation only")
        return False


def run_hardware_validation():
    """Run hardware validation experiments."""
    print("\n⚛️ HARDWARE VALIDATION")
    print("=" * 40)
    
    # Validate hardware setup
    print("Validating hardware configuration...")
    is_valid = validate_experimental_setup(HARDWARE_PROOF_OF_CONCEPT)
    print(f"Configuration valid: {is_valid}\n")
    
    # Initialize backend manager
    backend_manager = QuantumBackendManager()
    available_backends = backend_manager.list_backends()
    
    print("Available quantum backends:")
    for name, info in available_backends.items():
        qubits = info.get('n_qubits', 'Unknown')
        provider = info.get('provider', 'Unknown')
        is_sim = info.get('is_simulator', True)
        status = "Simulator" if is_sim else "Real Hardware"
        print(f"  - {name}: {qubits} qubits ({provider}) [{status}]")
        
    # Create minimal test dataset for hardware
    print(f"\nCreating minimal test dataset for hardware...")
    X_test = torch.randn(3, 5, 4)  # Very small for hardware costs
    y_test = torch.randint(0, 2, (3, 5))
    
    print(f"Test data: {X_test.shape} (minimal for cost control)")
    
    # Create hardware-optimized model
    model = ExperimentalQMANN(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        n_qubits=6,  # Small for hardware
        backend_manager=backend_manager
    )
    
    print(f"\nExperimental QMANN Model:")
    print(f"  - Input dim: 4")
    print(f"  - Hidden dim: 16") 
    print(f"  - Output dim: 2")
    print(f"  - Qubits: 6")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, X_test, y_test, backend_manager


def run_simulator_experiments(model, X_test, y_test):
    """Run experiments on quantum simulators (free)."""
    print(f"\n💻 SIMULATOR EXPERIMENTS (FREE)")
    print("=" * 50)
    
    simulator_results = {}
    
    # Test on different simulators
    simulators = ['ibm_simulator', 'google_simulator']
    
    for sim_name in simulators:
        print(f"\nTesting on {sim_name}...")
        
        try:
            start_time = time.time()
            
            # Run experimental forward pass
            output, exp_info = model.experimental_forward(
                X_test,
                backend_name=sim_name,
                log_experiment=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            simulator_results[sim_name] = {
                'success': exp_info['quantum_success'],
                'output_shape': output.shape,
                'execution_time': execution_time,
                'backend_info': exp_info.get('backend_info', {}),
                'hardware_stats': exp_info.get('hardware_stats', {})
            }
            
            print(f"  ✅ Success: {exp_info['quantum_success']}")
            print(f"  ✅ Output shape: {output.shape}")
            print(f"  ✅ Execution time: {execution_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            simulator_results[sim_name] = {'error': str(e)}
            
    return simulator_results


def run_hardware_experiments(model, X_test, y_test, use_real_hardware: bool = False):
    """Run experiments on real quantum hardware (paid)."""
    if not use_real_hardware:
        print(f"\n⚛️ REAL HARDWARE EXPERIMENTS SKIPPED")
        print("=" * 50)
        print("Real hardware experiments disabled for safety")
        print("Set use_real_hardware=True to enable (costs money!)")
        return {"message": "Hardware experiments skipped"}
        
    print(f"\n⚛️ REAL HARDWARE EXPERIMENTS (PAID)")
    print("=" * 50)
    print("🚨 WARNING: This will cost real money!")
    
    # Run hardware benchmark
    hardware_results = model.run_hardware_benchmark(
        test_data=X_test,
        backends=['ibm_brisbane']  # Start with one backend
    )
    
    return hardware_results


def analyze_hardware_results(simulator_results: Dict, hardware_results: Dict):
    """Analyze and compare hardware vs simulation results."""
    print(f"\n📊 RESULTS ANALYSIS")
    print("=" * 30)
    
    # Simulator analysis
    print("Simulator Results:")
    sim_success_rate = 0
    sim_count = 0
    
    for sim_name, result in simulator_results.items():
        if 'success' in result:
            success = result['success']
            exec_time = result.get('execution_time', 0)
            
            print(f"  {sim_name}:")
            print(f"    - Success: {'✅' if success else '❌'}")
            print(f"    - Execution time: {exec_time:.3f}s")
            
            if success:
                sim_success_rate += 1
            sim_count += 1
            
    if sim_count > 0:
        sim_success_rate = sim_success_rate / sim_count
        print(f"  Overall simulator success rate: {sim_success_rate:.2%}")
        
    # Hardware analysis
    print(f"\nHardware Results:")
    if hardware_results.get('message') == "Hardware experiments skipped":
        print("  Hardware experiments were skipped")
    else:
        for backend, result in hardware_results.items():
            if 'success_rate' in result:
                print(f"  {backend}:")
                print(f"    - Success rate: {result['success_rate']:.2%}")
                print(f"    - Avg execution time: {np.mean(result.get('execution_times', [0])):.3f}s")
                
    return {
        'simulator_success_rate': sim_success_rate,
        'hardware_results': hardware_results
    }


def generate_hardware_report(cost_estimate: Dict, simulator_results: Dict, 
                           hardware_results: Dict, analysis: Dict) -> str:
    """Generate comprehensive hardware experiment report."""
    
    report = []
    report.append("QMANN HARDWARE MODE REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"• Estimated cost: ${cost_estimate.get('total_cost', 0):.2f}")
    report.append(f"• Simulator success rate: {analysis['simulator_success_rate']:.2%}")
    
    if hardware_results.get('message') != "Hardware experiments skipped":
        report.append("• Real hardware experiments completed")
    else:
        report.append("• Real hardware experiments skipped (simulation only)")
        
    report.append("")
    
    # Cost Analysis
    report.append("COST ANALYSIS")
    report.append("-" * 15)
    report.append(f"• Qubits used: {cost_estimate.get('n_qubits', 'N/A')}")
    report.append(f"• Experiments: {cost_estimate.get('n_experiments', 'N/A')}")
    report.append(f"• Shots per experiment: {cost_estimate.get('shots_per_experiment', 'N/A')}")
    
    for backend, cost in cost_estimate.get('breakdown', {}).items():
        report.append(f"• {backend}: ${cost:.2f}")
        
    report.append("")
    
    # Technical Results
    report.append("TECHNICAL RESULTS")
    report.append("-" * 20)
    
    # Simulator results
    report.append("Simulator Performance:")
    for sim_name, result in simulator_results.items():
        if 'success' in result:
            status = "✅ Success" if result['success'] else "❌ Failed"
            exec_time = result.get('execution_time', 0)
            report.append(f"  - {sim_name}: {status} ({exec_time:.3f}s)")
            
    # Hardware results
    if hardware_results.get('message') != "Hardware experiments skipped":
        report.append("Hardware Performance:")
        for backend, result in hardware_results.items():
            if 'success_rate' in result:
                report.append(f"  - {backend}: {result['success_rate']:.2%} success rate")
    else:
        report.append("Hardware Performance: Experiments skipped")
        
    report.append("")
    
    # Limitations and Recommendations
    report.append("LIMITATIONS AND RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("• Current NISQ devices have high error rates (~1-10%)")
    report.append("• Limited to small qubit counts (4-12 qubits)")
    report.append("• Real quantum advantage requires fault-tolerant devices")
    report.append("• Cost-effective for proof-of-concept experiments only")
    report.append("")
    report.append("Recommendations:")
    report.append("• Start with simulator validation (free)")
    report.append("• Use minimal test cases for hardware (cost control)")
    report.append("• Focus on algorithm validation rather than performance")
    report.append("• Plan for fault-tolerant quantum computers (2030s)")
    
    return "\n".join(report)


def main():
    """Main hardware experiment workflow."""
    print("⚛️ QMANN HARDWARE MODE EXPERIMENTS")
    print("=" * 50)
    print("Real quantum hardware validation")
    print("⚠️  WARNING: This mode can cost real money!")
    print()
    
    # Configuration
    USE_REAL_HARDWARE = False  # Set to True for real hardware (costs money!)
    N_QUBITS = 6
    N_EXPERIMENTS = 3
    SHOTS_PER_EXPERIMENT = 512  # Reduced for cost control
    
    try:
        # Step 1: Check prerequisites
        if not check_hardware_prerequisites():
            return
            
        # Step 2: Estimate costs
        cost_estimate = estimate_experiment_costs(N_QUBITS, N_EXPERIMENTS, SHOTS_PER_EXPERIMENT)
        
        # Step 3: Safety confirmation
        if not safety_confirmation(cost_estimate):
            USE_REAL_HARDWARE = False
            
        # Step 4: Setup hardware validation
        model, X_test, y_test, backend_manager = run_hardware_validation()
        
        # Step 5: Run simulator experiments (always free)
        simulator_results = run_simulator_experiments(model, X_test, y_test)
        
        # Step 6: Run hardware experiments (if approved)
        hardware_results = run_hardware_experiments(model, X_test, y_test, USE_REAL_HARDWARE)
        
        # Step 7: Analyze results
        analysis = analyze_hardware_results(simulator_results, hardware_results)
        
        # Step 8: Generate report
        report = generate_hardware_report(cost_estimate, simulator_results, hardware_results, analysis)
        
        print("\n" + "=" * 50)
        print("HARDWARE EXPERIMENTS COMPLETE")
        print("=" * 50)
        print(report)
        
        # Save report
        with open('qmann_hardware_report.txt', 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: qmann_hardware_report.txt")
        
        # Save experimental summary
        experimental_summary = model.get_experimental_summary()
        print(f"\n📊 Experimental Summary:")
        for key, value in experimental_summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Hardware experiments failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
