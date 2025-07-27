#!/usr/bin/env python3
"""
Quantum Hardware Cost Estimation Tool

This script estimates the cost of running QMANN experiments on real quantum hardware
before actually executing them, helping researchers budget their experiments.
"""

import argparse
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BackendCosts:
    """Cost structure for quantum backends."""
    name: str
    provider: str
    cost_per_shot: float
    cost_per_second: float
    max_qubits: int
    free_tier_shots: int = 0
    notes: str = ""


# 2025 Quantum Hardware Pricing (approximate)
BACKEND_COSTS = {
    'ibm_brisbane': BackendCosts(
        name="IBM Brisbane",
        provider="IBM Quantum",
        cost_per_shot=0.001,
        cost_per_second=1.60,
        max_qubits=127,
        free_tier_shots=1000,  # IBM Quantum Network free tier
        notes="Superconducting, high connectivity"
    ),
    'ibm_kyoto': BackendCosts(
        name="IBM Kyoto", 
        provider="IBM Quantum",
        cost_per_shot=0.001,
        cost_per_second=1.60,
        max_qubits=127,
        free_tier_shots=1000,
        notes="Superconducting, latest generation"
    ),
    'ionq_aria': BackendCosts(
        name="IonQ Aria",
        provider="IonQ",
        cost_per_shot=0.01,
        cost_per_second=0.0,  # IonQ charges per shot
        max_qubits=25,
        free_tier_shots=100,  # Limited free tier
        notes="Trapped ion, all-to-all connectivity"
    ),
    'google_sycamore': BackendCosts(
        name="Google Sycamore",
        provider="Google Quantum AI",
        cost_per_shot=0.005,
        cost_per_second=2.00,
        max_qubits=70,
        free_tier_shots=0,  # No free tier
        notes="Superconducting, research access only"
    ),
    'aws_sv1': BackendCosts(
        name="AWS Braket SV1",
        provider="AWS Braket",
        cost_per_shot=0.075,  # Per task + per shot
        cost_per_second=0.0,
        max_qubits=34,
        free_tier_shots=0,
        notes="State vector simulator"
    ),
    'rigetti_aspen': BackendCosts(
        name="Rigetti Aspen-M",
        provider="Rigetti",
        cost_per_shot=0.00035,
        cost_per_second=0.0,
        max_qubits=80,
        free_tier_shots=0,
        notes="Superconducting, forest SDK"
    )
}

# Simulator costs (free)
SIMULATOR_COSTS = {
    'ibm_simulator': BackendCosts(
        name="IBM Qiskit Aer",
        provider="IBM (Local)",
        cost_per_shot=0.0,
        cost_per_second=0.0,
        max_qubits=32,
        free_tier_shots=float('inf'),
        notes="Free local simulation"
    ),
    'google_simulator': BackendCosts(
        name="Google Cirq",
        provider="Google (Local)",
        cost_per_shot=0.0,
        cost_per_second=0.0,
        max_qubits=30,
        free_tier_shots=float('inf'),
        notes="Free local simulation"
    )
}


def estimate_experiment_cost(backend_name: str, n_qubits: int, shots: int, 
                           circuit_depth: int = 50, n_experiments: int = 1) -> Dict:
    """
    Estimate cost for quantum experiment.
    
    Args:
        backend_name: Name of quantum backend
        n_qubits: Number of qubits required
        shots: Number of measurement shots
        circuit_depth: Estimated circuit depth
        n_experiments: Number of experiments to run
        
    Returns:
        Cost estimation dictionary
    """
    # Get backend costs
    if backend_name in BACKEND_COSTS:
        backend = BACKEND_COSTS[backend_name]
    elif backend_name in SIMULATOR_COSTS:
        backend = SIMULATOR_COSTS[backend_name]
    else:
        return {"error": f"Unknown backend: {backend_name}"}
        
    # Check qubit availability
    if n_qubits > backend.max_qubits:
        return {
            "error": f"Requested {n_qubits} qubits exceeds {backend.name} limit of {backend.max_qubits}"
        }
        
    # Calculate costs
    total_shots = shots * n_experiments
    
    # Shot-based cost
    shot_cost = 0.0
    if total_shots > backend.free_tier_shots:
        paid_shots = total_shots - backend.free_tier_shots
        shot_cost = paid_shots * backend.cost_per_shot
        
    # Time-based cost (estimate)
    estimated_runtime_seconds = circuit_depth * 0.1 * n_experiments  # Rough estimate
    time_cost = estimated_runtime_seconds * backend.cost_per_second
    
    # Total cost (use higher of shot-based or time-based)
    total_cost = max(shot_cost, time_cost)
    
    return {
        "backend": backend.name,
        "provider": backend.provider,
        "n_qubits": n_qubits,
        "total_shots": total_shots,
        "free_shots_used": min(total_shots, backend.free_tier_shots),
        "paid_shots": max(0, total_shots - backend.free_tier_shots),
        "shot_cost": shot_cost,
        "time_cost": time_cost,
        "total_cost": total_cost,
        "estimated_runtime_seconds": estimated_runtime_seconds,
        "notes": backend.notes
    }


def estimate_qmann_experiment_cost(n_qubits: int, memory_capacity: int, 
                                 training_epochs: int = 10, 
                                 backends: List[str] = None) -> Dict:
    """
    Estimate cost for complete QMANN experiment.
    
    Args:
        n_qubits: Number of qubits for quantum components
        memory_capacity: Quantum memory capacity
        training_epochs: Number of training epochs
        backends: List of backends to estimate for
        
    Returns:
        Complete cost estimation
    """
    if backends is None:
        backends = ['ibm_brisbane', 'ionq_aria']
        
    # QMANN experiment parameters
    shots_per_forward = 1024  # Standard shots for measurement
    forwards_per_epoch = memory_capacity // 4  # Rough estimate
    total_forwards = forwards_per_epoch * training_epochs
    circuit_depth = min(50, n_qubits * 5)  # NISQ depth limit
    
    results = {
        "experiment_parameters": {
            "n_qubits": n_qubits,
            "memory_capacity": memory_capacity,
            "training_epochs": training_epochs,
            "total_forward_passes": total_forwards,
            "shots_per_forward": shots_per_forward,
            "total_shots": total_forwards * shots_per_forward,
            "estimated_circuit_depth": circuit_depth
        },
        "backend_estimates": {}
    }
    
    total_cost_all_backends = 0.0
    
    for backend_name in backends:
        estimate = estimate_experiment_cost(
            backend_name=backend_name,
            n_qubits=n_qubits,
            shots=shots_per_forward,
            circuit_depth=circuit_depth,
            n_experiments=total_forwards
        )
        
        results["backend_estimates"][backend_name] = estimate
        
        if "error" not in estimate:
            total_cost_all_backends += estimate["total_cost"]
            
    results["total_cost_all_backends"] = total_cost_all_backends
    
    return results


def print_cost_estimate(estimate: Dict, detailed: bool = False):
    """Print formatted cost estimate."""
    if "error" in estimate:
        print(f"❌ Error: {estimate['error']}")
        return
        
    print(f"💰 Cost Estimate: {estimate['backend']} ({estimate['provider']})")
    print(f"   Qubits: {estimate['n_qubits']}")
    print(f"   Total shots: {estimate['total_shots']:,}")
    
    if estimate['free_shots_used'] > 0:
        print(f"   Free shots: {estimate['free_shots_used']:,}")
        
    if estimate['paid_shots'] > 0:
        print(f"   Paid shots: {estimate['paid_shots']:,}")
        print(f"   Shot cost: ${estimate['shot_cost']:.2f}")
        
    if estimate['time_cost'] > 0:
        print(f"   Time cost: ${estimate['time_cost']:.2f}")
        print(f"   Runtime: {estimate['estimated_runtime_seconds']:.1f}s")
        
    print(f"   💵 TOTAL COST: ${estimate['total_cost']:.2f}")
    
    if detailed:
        print(f"   Notes: {estimate['notes']}")
        
    print()


def main():
    """Main cost estimation function."""
    parser = argparse.ArgumentParser(description="Estimate quantum hardware costs for QMANN experiments")
    
    parser.add_argument("--qubits", type=int, default=6, help="Number of qubits (default: 6)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per experiment (default: 1024)")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments (default: 1)")
    parser.add_argument("--backends", nargs="+", default=["ibm_brisbane", "ionq_aria"], 
                       help="Backends to estimate (default: ibm_brisbane ionq_aria)")
    parser.add_argument("--qmann-experiment", action="store_true", 
                       help="Estimate full QMANN training experiment")
    parser.add_argument("--memory-capacity", type=int, default=32, 
                       help="QMANN memory capacity (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Training epochs for QMANN experiment (default: 10)")
    parser.add_argument("--list-backends", action="store_true", 
                       help="List available backends and exit")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    if args.list_backends:
        print("Available Quantum Hardware Backends:")
        print("=" * 50)
        
        for name, backend in BACKEND_COSTS.items():
            print(f"🔧 {name}")
            print(f"   Provider: {backend.provider}")
            print(f"   Max qubits: {backend.max_qubits}")
            print(f"   Cost per shot: ${backend.cost_per_shot:.4f}")
            if backend.cost_per_second > 0:
                print(f"   Cost per second: ${backend.cost_per_second:.2f}")
            if backend.free_tier_shots > 0:
                print(f"   Free tier: {backend.free_tier_shots:,} shots")
            print(f"   Notes: {backend.notes}")
            print()
            
        print("Available Simulators (Free):")
        print("=" * 30)
        
        for name, backend in SIMULATOR_COSTS.items():
            print(f"💻 {name}")
            print(f"   Provider: {backend.provider}")
            print(f"   Max qubits: {backend.max_qubits}")
            print(f"   Cost: FREE")
            print(f"   Notes: {backend.notes}")
            print()
            
        return
        
    print("QMANN Quantum Hardware Cost Estimator")
    print("=" * 50)
    
    if args.qmann_experiment:
        print(f"Estimating QMANN training experiment:")
        print(f"  - Qubits: {args.qubits}")
        print(f"  - Memory capacity: {args.memory_capacity}")
        print(f"  - Training epochs: {args.epochs}")
        print(f"  - Backends: {', '.join(args.backends)}")
        print()
        
        estimate = estimate_qmann_experiment_cost(
            n_qubits=args.qubits,
            memory_capacity=args.memory_capacity,
            training_epochs=args.epochs,
            backends=args.backends
        )
        
        params = estimate["experiment_parameters"]
        print("📊 Experiment Parameters:")
        print(f"   Total forward passes: {params['total_forward_passes']:,}")
        print(f"   Total shots: {params['total_shots']:,}")
        print(f"   Circuit depth: {params['estimated_circuit_depth']}")
        print()
        
        print("💰 Cost Estimates by Backend:")
        print("-" * 40)
        
        for backend_name, backend_estimate in estimate["backend_estimates"].items():
            print_cost_estimate(backend_estimate, args.detailed)
            
        if len(estimate["backend_estimates"]) > 1:
            print(f"💵 TOTAL COST (ALL BACKENDS): ${estimate['total_cost_all_backends']:.2f}")
            
    else:
        print(f"Estimating single experiment:")
        print(f"  - Qubits: {args.qubits}")
        print(f"  - Shots: {args.shots}")
        print(f"  - Experiments: {args.experiments}")
        print(f"  - Backends: {', '.join(args.backends)}")
        print()
        
        total_cost = 0.0
        
        for backend_name in args.backends:
            estimate = estimate_experiment_cost(
                backend_name=backend_name,
                n_qubits=args.qubits,
                shots=args.shots,
                n_experiments=args.experiments
            )
            
            print_cost_estimate(estimate, args.detailed)
            
            if "error" not in estimate:
                total_cost += estimate["total_cost"]
                
        if len(args.backends) > 1:
            print(f"💵 TOTAL COST (ALL BACKENDS): ${total_cost:.2f}")
            
    # Cost warnings
    if total_cost > 100:
        print("⚠️  WARNING: High cost experiment (>$100)")
        print("   Consider reducing qubits, shots, or experiments")
    elif total_cost > 10:
        print("⚠️  CAUTION: Moderate cost experiment (>$10)")
        print("   Ensure budget approval before proceeding")
    elif total_cost > 0:
        print("✅ Low cost experiment (<$10)")
    else:
        print("🆓 FREE experiment (simulators only)")


if __name__ == "__main__":
    main()
