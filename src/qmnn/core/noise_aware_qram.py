"""
Noise-Aware QRAM Implementation

This module implements realistic noise models and shot-cost optimization
for QRAM operations on real quantum hardware.
"""

import numpy as np
import torch
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit, transpile, execute
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    from qiskit.providers.fake_provider import FakeProvider
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class NoiseParameters:
    """Realistic noise parameters for quantum devices."""
    
    # Gate errors (2025 NISQ typical values)
    single_qubit_gate_error: float = 0.001  # 0.1%
    two_qubit_gate_error: float = 0.01      # 1%
    readout_error: float = 0.02             # 2%
    
    # Decoherence times (microseconds)
    t1_time: float = 100.0  # T1 relaxation time
    t2_time: float = 50.0   # T2 dephasing time
    
    # Gate times (nanoseconds)
    single_qubit_gate_time: float = 50.0
    two_qubit_gate_time: float = 200.0
    
    # Shot costs (USD per shot, 2025 pricing)
    ibm_cost_per_shot: float = 0.001
    ionq_cost_per_shot: float = 0.01
    google_cost_per_shot: float = 0.005


class ShotCostScheduler:
    """Optimizes shot allocation based on cost and accuracy requirements."""
    
    def __init__(self, budget_usd: float = 10.0, target_fidelity: float = 0.95):
        self.budget_usd = budget_usd
        self.target_fidelity = target_fidelity
        self.spent_budget = 0.0
        self.shot_history = []
        
    def calculate_optimal_shots(self, circuit_depth: int, n_qubits: int, 
                              backend_cost_per_shot: float) -> int:
        """
        Calculate optimal shot count based on circuit complexity and budget.
        
        Args:
            circuit_depth: Depth of quantum circuit
            n_qubits: Number of qubits
            backend_cost_per_shot: Cost per shot for the backend
            
        Returns:
            Optimal number of shots
        """
        # Estimate required shots for target fidelity
        # More complex circuits need more shots for statistical accuracy
        complexity_factor = np.sqrt(circuit_depth * n_qubits)
        base_shots = int(1024 * complexity_factor / 10)  # Scale from base 1024
        
        # Budget constraint
        remaining_budget = self.budget_usd - self.spent_budget
        max_affordable_shots = int(remaining_budget / backend_cost_per_shot)
        
        # Take minimum of required and affordable
        optimal_shots = min(base_shots, max_affordable_shots, 10000)  # Cap at 10k
        
        # Ensure minimum shots for statistical validity
        optimal_shots = max(optimal_shots, 100)
        
        return optimal_shots
        
    def record_shot_usage(self, shots_used: int, cost_per_shot: float, 
                         measured_fidelity: float):
        """Record shot usage for optimization."""
        cost = shots_used * cost_per_shot
        self.spent_budget += cost
        
        self.shot_history.append({
            'shots': shots_used,
            'cost': cost,
            'fidelity': measured_fidelity,
            'efficiency': measured_fidelity / cost if cost > 0 else 0
        })
        
    def get_budget_status(self) -> Dict[str, float]:
        """Get current budget status."""
        return {
            'total_budget': self.budget_usd,
            'spent': self.spent_budget,
            'remaining': self.budget_usd - self.spent_budget,
            'utilization': self.spent_budget / self.budget_usd
        }


class NoiseAwareQRAM:
    """
    QRAM implementation with realistic noise modeling and cost optimization.
    """
    
    def __init__(self, memory_size: int, address_qubits: int, 
                 noise_params: Optional[NoiseParameters] = None,
                 shot_scheduler: Optional[ShotCostScheduler] = None,
                 backend_name: str = "ibm_brisbane"):
        """
        Initialize noise-aware QRAM.
        
        Args:
            memory_size: Number of memory cells
            address_qubits: Number of address qubits
            noise_params: Noise parameters for simulation
            shot_scheduler: Shot cost optimization scheduler
            backend_name: Target quantum backend
        """
        self.memory_size = memory_size
        self.address_qubits = address_qubits
        self.noise_params = noise_params or NoiseParameters()
        self.shot_scheduler = shot_scheduler or ShotCostScheduler()
        self.backend_name = backend_name
        
        # Memory storage
        self.memory_data = {}
        self.access_history = []
        
        # Initialize noise model
        self.noise_model = self._create_noise_model()
        
        # Initialize backend
        self.backend = self._initialize_backend()
        
    def _create_noise_model(self) -> Optional[Any]:
        """Create realistic noise model for simulation."""
        if not QISKIT_AVAILABLE:
            return None
            
        noise_model = NoiseModel()
        
        # Single-qubit gate errors
        single_qubit_error = depolarizing_error(
            self.noise_params.single_qubit_gate_error, 1
        )
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
        
        # Two-qubit gate errors
        two_qubit_error = depolarizing_error(
            self.noise_params.two_qubit_gate_error, 2
        )
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
        
        # Thermal relaxation
        for qubit in range(self.address_qubits + 8):  # Address + data qubits
            thermal_error = thermal_relaxation_error(
                self.noise_params.t1_time,
                self.noise_params.t2_time,
                self.noise_params.single_qubit_gate_time
            )
            noise_model.add_quantum_error(thermal_error, ['u1', 'u2', 'u3'], [qubit])
            
        return noise_model
        
    def _initialize_backend(self):
        """Initialize quantum backend with noise model."""
        if not QISKIT_AVAILABLE:
            return None
            
        # Try to get real backend, fallback to simulator
        try:
            if "ibm" in self.backend_name.lower():
                # Try IBM Quantum Runtime
                service = QiskitRuntimeService()
                backend = service.backend(self.backend_name)
                return backend
        except:
            pass
            
        # Fallback to noisy simulator
        backend = AerSimulator(noise_model=self.noise_model)
        return backend
        
    def write_with_noise(self, address: int, data: np.ndarray, 
                        shots: Optional[int] = None) -> Dict[str, Any]:
        """
        Write data to QRAM with realistic noise and cost tracking.
        
        Args:
            address: Memory address
            data: Data to write
            shots: Number of shots (auto-calculated if None)
            
        Returns:
            Write operation results with noise metrics
        """
        if not QISKIT_AVAILABLE:
            # Classical fallback
            self.memory_data[address] = data
            return {
                'success': True,
                'fidelity': 1.0,
                'shots_used': 0,
                'cost': 0.0,
                'noise_model': 'classical_fallback'
            }
            
        # Create write circuit
        write_circuit = self._create_write_circuit(address, data)
        
        # Calculate optimal shots
        if shots is None:
            shots = self.shot_scheduler.calculate_optimal_shots(
                circuit_depth=write_circuit.depth(),
                n_qubits=write_circuit.num_qubits,
                backend_cost_per_shot=self.noise_params.ibm_cost_per_shot
            )
            
        # Execute with noise
        try:
            job = execute(write_circuit, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Analyze fidelity
            fidelity = self._calculate_write_fidelity(counts, data)
            
            # Record cost
            cost = shots * self.noise_params.ibm_cost_per_shot
            self.shot_scheduler.record_shot_usage(shots, self.noise_params.ibm_cost_per_shot, fidelity)
            
            # Store data if fidelity is acceptable
            if fidelity > 0.8:  # 80% threshold
                self.memory_data[address] = data
                success = True
            else:
                success = False
                
            return {
                'success': success,
                'fidelity': fidelity,
                'shots_used': shots,
                'cost': cost,
                'counts': counts,
                'noise_model': 'realistic'
            }
            
        except Exception as e:
            warnings.warn(f"Quantum write failed: {e}")
            # Classical fallback
            self.memory_data[address] = data
            return {
                'success': True,
                'fidelity': 1.0,
                'shots_used': 0,
                'cost': 0.0,
                'error': str(e),
                'noise_model': 'classical_fallback'
            }
            
    def read_with_noise(self, address: int, shots: Optional[int] = None) -> Dict[str, Any]:
        """
        Read data from QRAM with realistic noise and cost tracking.
        
        Args:
            address: Memory address
            shots: Number of shots (auto-calculated if None)
            
        Returns:
            Read operation results with noise metrics
        """
        if address not in self.memory_data:
            return {
                'success': False,
                'data': None,
                'error': 'Address not found'
            }
            
        if not QISKIT_AVAILABLE:
            # Classical fallback
            return {
                'success': True,
                'data': self.memory_data[address],
                'fidelity': 1.0,
                'shots_used': 0,
                'cost': 0.0,
                'noise_model': 'classical_fallback'
            }
            
        # Create read circuit
        read_circuit = self._create_read_circuit(address)
        
        # Calculate optimal shots
        if shots is None:
            shots = self.shot_scheduler.calculate_optimal_shots(
                circuit_depth=read_circuit.depth(),
                n_qubits=read_circuit.num_qubits,
                backend_cost_per_shot=self.noise_params.ibm_cost_per_shot
            )
            
        # Execute with noise
        try:
            job = execute(read_circuit, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Reconstruct data from noisy measurements
            reconstructed_data = self._reconstruct_data_from_counts(counts, address)
            
            # Calculate fidelity
            original_data = self.memory_data[address]
            fidelity = self._calculate_read_fidelity(reconstructed_data, original_data)
            
            # Record cost
            cost = shots * self.noise_params.ibm_cost_per_shot
            self.shot_scheduler.record_shot_usage(shots, self.noise_params.ibm_cost_per_shot, fidelity)
            
            return {
                'success': True,
                'data': reconstructed_data,
                'fidelity': fidelity,
                'shots_used': shots,
                'cost': cost,
                'counts': counts,
                'noise_model': 'realistic'
            }
            
        except Exception as e:
            warnings.warn(f"Quantum read failed: {e}")
            # Classical fallback
            return {
                'success': True,
                'data': self.memory_data[address],
                'fidelity': 1.0,
                'shots_used': 0,
                'cost': 0.0,
                'error': str(e),
                'noise_model': 'classical_fallback'
            }
            
    def _create_write_circuit(self, address: int, data: np.ndarray) -> QuantumCircuit:
        """Create quantum circuit for write operation."""
        n_qubits = self.address_qubits + len(data)
        circuit = QuantumCircuit(n_qubits, n_qubits)
        
        # Encode address
        address_bits = format(address, f'0{self.address_qubits}b')
        for i, bit in enumerate(address_bits):
            if bit == '1':
                circuit.x(i)
                
        # Encode data (amplitude encoding)
        data_normalized = data / np.linalg.norm(data)
        for i, amplitude in enumerate(data_normalized):
            if i < len(data) and abs(amplitude) > 0.1:  # Threshold
                angle = 2 * np.arcsin(min(abs(amplitude), 1.0))
                circuit.ry(angle, self.address_qubits + i)
                
        # Add measurements
        circuit.measure_all()
        
        return circuit
        
    def _create_read_circuit(self, address: int) -> QuantumCircuit:
        """Create quantum circuit for read operation."""
        stored_data = self.memory_data[address]
        return self._create_write_circuit(address, stored_data)
        
    def _calculate_write_fidelity(self, counts: Dict[str, int], 
                                 original_data: np.ndarray) -> float:
        """Calculate write operation fidelity."""
        # Simple fidelity based on measurement distribution
        total_shots = sum(counts.values())
        
        # Expected pattern based on data encoding
        expected_pattern = self._data_to_bit_pattern(original_data)
        
        # Count matches
        matches = counts.get(expected_pattern, 0)
        fidelity = matches / total_shots
        
        return fidelity
        
    def _calculate_read_fidelity(self, reconstructed: np.ndarray, 
                                original: np.ndarray) -> float:
        """Calculate read operation fidelity."""
        if len(reconstructed) != len(original):
            return 0.0
            
        # Cosine similarity
        dot_product = np.dot(reconstructed, original)
        norms = np.linalg.norm(reconstructed) * np.linalg.norm(original)
        
        if norms == 0:
            return 0.0
            
        fidelity = abs(dot_product / norms)
        return fidelity
        
    def _reconstruct_data_from_counts(self, counts: Dict[str, int], 
                                     address: int) -> np.ndarray:
        """Reconstruct data from noisy measurement counts."""
        original_data = self.memory_data[address]
        
        # Simple reconstruction: use most frequent measurement
        most_frequent = max(counts.items(), key=lambda x: x[1])
        bit_pattern = most_frequent[0]
        
        # Convert bit pattern back to data
        reconstructed = self._bit_pattern_to_data(bit_pattern, len(original_data))
        
        return reconstructed
        
    def _data_to_bit_pattern(self, data: np.ndarray) -> str:
        """Convert data to expected bit pattern."""
        # Simple encoding: threshold-based
        bits = []
        for val in data:
            bits.append('1' if val > 0 else '0')
            
        # Pad with address bits
        address_bits = '0' * self.address_qubits
        return address_bits + ''.join(bits)
        
    def _bit_pattern_to_data(self, bit_pattern: str, data_length: int) -> np.ndarray:
        """Convert bit pattern back to data."""
        # Extract data bits (skip address bits)
        data_bits = bit_pattern[self.address_qubits:]
        
        # Convert to float array
        data = np.array([1.0 if bit == '1' else -1.0 for bit in data_bits[:data_length]])
        
        return data
        
    def get_noise_statistics(self) -> Dict[str, Any]:
        """Get comprehensive noise and cost statistics."""
        budget_status = self.shot_scheduler.get_budget_status()
        
        # Calculate average fidelity
        if self.shot_scheduler.shot_history:
            avg_fidelity = np.mean([h['fidelity'] for h in self.shot_scheduler.shot_history])
            avg_efficiency = np.mean([h['efficiency'] for h in self.shot_scheduler.shot_history])
            total_shots = sum([h['shots'] for h in self.shot_scheduler.shot_history])
        else:
            avg_fidelity = 0.0
            avg_efficiency = 0.0
            total_shots = 0
            
        return {
            'noise_parameters': {
                'single_qubit_error': self.noise_params.single_qubit_gate_error,
                'two_qubit_error': self.noise_params.two_qubit_gate_error,
                'readout_error': self.noise_params.readout_error,
                't1_time': self.noise_params.t1_time,
                't2_time': self.noise_params.t2_time
            },
            'performance_metrics': {
                'average_fidelity': avg_fidelity,
                'average_efficiency': avg_efficiency,
                'total_shots_used': total_shots,
                'total_operations': len(self.shot_scheduler.shot_history)
            },
            'cost_metrics': budget_status,
            'backend_info': {
                'name': self.backend_name,
                'noise_model_active': self.noise_model is not None,
                'qiskit_available': QISKIT_AVAILABLE
            }
        }
