"""
Utility functions for QMNN.

This module provides helper functions for quantum state manipulation,
data preprocessing, and visualization.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.visualization import plot_circuit_layout, plot_histogram
from qiskit.quantum_info import Statevector
import pandas as pd


def quantum_state_to_classical(state: Statevector, method: str = "expectation") -> np.ndarray:
    """
    Convert quantum state to classical representation.
    
    Args:
        state: Quantum state vector
        method: Conversion method ('expectation', 'measurement', 'amplitude')
        
    Returns:
        Classical vector representation
    """
    if method == "expectation":
        # Use expectation values of Pauli-Z operators
        n_qubits = int(np.log2(len(state.data)))
        classical = np.zeros(n_qubits)
        
        for i in range(n_qubits):
            # Compute <Z_i> expectation value
            classical[i] = np.real(state.expectation_value(f"Z_{i}"))
            
    elif method == "measurement":
        # Simulate measurement outcomes
        probabilities = np.abs(state.data) ** 2
        classical = np.random.choice(len(probabilities), p=probabilities)
        classical = np.array([int(x) for x in format(classical, f'0{int(np.log2(len(probabilities)))}b')])
        
    elif method == "amplitude":
        # Use real parts of amplitudes
        classical = np.real(state.data)
        
    else:
        raise ValueError(f"Unknown conversion method: {method}")
        
    return classical


def classical_to_quantum_state(classical: np.ndarray, encoding: str = "amplitude") -> Statevector:
    """
    Convert classical data to quantum state.
    
    Args:
        classical: Classical data vector
        encoding: Encoding method ('amplitude', 'angle', 'basis')
        
    Returns:
        Quantum state vector
    """
    if encoding == "amplitude":
        # Normalize and use as amplitudes
        normalized = classical / np.linalg.norm(classical) if np.linalg.norm(classical) > 0 else classical
        
        # Pad to power of 2 if necessary
        n_qubits = int(np.ceil(np.log2(len(normalized))))
        padded_size = 2 ** n_qubits
        
        if len(normalized) < padded_size:
            padded = np.zeros(padded_size)
            padded[:len(normalized)] = normalized
            normalized = padded
            
        state = Statevector(normalized)
        
    elif encoding == "angle":
        # Encode as rotation angles
        n_qubits = len(classical)
        qc = QuantumCircuit(n_qubits)
        
        for i, angle in enumerate(classical):
            qc.ry(angle * np.pi, i)  # Scale to [0, π]
            
        state = Statevector.from_instruction(qc)
        
    elif encoding == "basis":
        # Encode as computational basis state
        n_qubits = int(np.ceil(np.log2(len(classical))))
        basis_index = int(''.join([str(int(x > 0.5)) for x in classical[:n_qubits]]), 2)
        
        state_vector = np.zeros(2 ** n_qubits)
        state_vector[basis_index] = 1.0
        state = Statevector(state_vector)
        
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")
        
    return state


def visualize_quantum_circuit(circuit: QuantumCircuit, save_path: Optional[str] = None) -> None:
    """
    Visualize quantum circuit.
    
    Args:
        circuit: Quantum circuit to visualize
        save_path: Optional path to save figure
    """
    fig = circuit.draw(output='mpl', style='iqx')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['train_losses'], label='Train Loss', color='blue')
    if 'val_losses' in history and history['val_losses']:
        axes[0].plot(history['val_losses'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_accuracies'], label='Train Acc', color='blue')
    if 'val_accuracies' in history and history['val_accuracies']:
        axes[1].plot(history['val_accuracies'], label='Val Acc', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_memory_usage(
    memory_usage_history: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4)
) -> None:
    """
    Plot quantum memory usage over time.
    
    Args:
        memory_usage_history: List of memory usage ratios
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(memory_usage_history, color='green', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Memory Usage Ratio')
    plt.title('Quantum Memory Usage During Training')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def create_benchmark_plot(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create benchmark comparison plot.
    
    Args:
        results_df: DataFrame with benchmark results
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy comparison
    sns.barplot(data=results_df, x='model', y='accuracy', ax=axes[0])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Runtime comparison
    sns.barplot(data=results_df, x='model', y='runtime_ms', ax=axes[1])
    axes[1].set_title('Model Runtime Comparison')
    axes[1].set_ylabel('Runtime (ms)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def validate_quantum_state(state: Statevector, tolerance: float = 1e-10) -> bool:
    """
    Validate quantum state normalization.
    
    Args:
        state: Quantum state to validate
        tolerance: Numerical tolerance
        
    Returns:
        True if state is properly normalized
    """
    norm_squared = np.sum(np.abs(state.data) ** 2)
    return abs(norm_squared - 1.0) < tolerance


def compute_quantum_fidelity(state1: Statevector, state2: Statevector) -> float:
    """
    Compute fidelity between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity value between 0 and 1
    """
    overlap = np.abs(np.vdot(state1.data, state2.data)) ** 2
    return overlap


def generate_random_quantum_data(
    n_samples: int,
    n_qubits: int,
    seed: Optional[int] = None
) -> List[Statevector]:
    """
    Generate random quantum states for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_qubits: Number of qubits per state
        seed: Random seed
        
    Returns:
        List of random quantum states
    """
    if seed is not None:
        np.random.seed(seed)
        
    states = []
    dim = 2 ** n_qubits
    
    for _ in range(n_samples):
        # Generate random complex amplitudes
        real_part = np.random.randn(dim)
        imag_part = np.random.randn(dim)
        amplitudes = real_part + 1j * imag_part
        
        # Normalize
        amplitudes /= np.linalg.norm(amplitudes)
        
        states.append(Statevector(amplitudes))
        
    return states


def save_quantum_circuit_qasm(circuit: QuantumCircuit, filepath: str) -> None:
    """
    Save quantum circuit as QASM file.
    
    Args:
        circuit: Quantum circuit to save
        filepath: Path to save QASM file
    """
    qasm_str = circuit.qasm()
    
    with open(filepath, 'w') as f:
        f.write(qasm_str)


def load_quantum_circuit_qasm(filepath: str) -> QuantumCircuit:
    """
    Load quantum circuit from QASM file.
    
    Args:
        filepath: Path to QASM file
        
    Returns:
        Loaded quantum circuit
    """
    return QuantumCircuit.from_qasm_file(filepath)


def estimate_quantum_resources(circuit: QuantumCircuit) -> dict:
    """
    Estimate quantum resources required for circuit.
    
    Args:
        circuit: Quantum circuit to analyze
        
    Returns:
        Dictionary with resource estimates
    """
    return {
        'n_qubits': circuit.num_qubits,
        'n_classical_bits': circuit.num_clbits,
        'depth': circuit.depth(),
        'gate_count': len(circuit.data),
        'cx_count': circuit.count_ops().get('cx', 0),
        'single_qubit_gates': sum(1 for gate, _, _ in circuit.data if len(gate.qubits) == 1),
        'two_qubit_gates': sum(1 for gate, _, _ in circuit.data if len(gate.qubits) == 2),
    }
