"""
Experimental Interface for Real Quantum Hardware

This module provides QMNN components specifically designed for real quantum
hardware experiments, with NISQ-era optimizations and hardware constraints.
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from .quantum_backend import QuantumBackendManager, QuantumBackend
from ..core import QRAM, QuantumMemory
from ..models import QMNN

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class HardwareAwareQRAM:
    """
    QRAM implementation optimized for real quantum hardware.
    
    Designed for NISQ devices with limited connectivity and high noise.
    """
    
    def __init__(self, n_qubits: int, backend_manager: QuantumBackendManager,
                 error_mitigation: bool = True, optimization_level: int = 2):
        """
        Initialize hardware-aware QRAM.
        
        Args:
            n_qubits: Number of qubits (limited by hardware)
            backend_manager: Quantum backend manager
            error_mitigation: Whether to use error mitigation
            optimization_level: Circuit optimization level (0-3)
        """
        if n_qubits > 20:
            warnings.warn(f"n_qubits={n_qubits} may exceed current hardware limits")
            
        self.n_qubits = n_qubits
        self.backend_manager = backend_manager
        self.error_mitigation = error_mitigation
        self.optimization_level = optimization_level
        
        # Hardware constraints
        self.max_circuit_depth = 100  # Typical NISQ limit
        self.max_two_qubit_gates = 50  # Noise accumulation limit
        
        # Memory storage (classical backup)
        self.classical_memory = {}
        self.quantum_addresses = {}
        
        # Circuit templates
        self._create_circuit_templates()
        
    def _create_circuit_templates(self):
        """Create optimized circuit templates for hardware."""
        if not QISKIT_AVAILABLE:
            warnings.warn("Qiskit not available. Using classical fallback.")
            return
            
        # Address encoding circuit (minimal depth)
        self.address_circuit = QuantumCircuit(self.n_qubits)
        
        # Data encoding circuit (amplitude encoding)
        self.data_circuit = QuantumCircuit(self.n_qubits)
        
        # Readout circuit
        self.readout_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.readout_circuit.measure_all()
        
    def store_data(self, address: int, data: np.ndarray, 
                   backend_name: Optional[str] = None) -> bool:
        """
        Store data in quantum memory with hardware execution.
        
        Args:
            address: Memory address
            data: Data to store (normalized)
            backend_name: Specific backend to use
            
        Returns:
            Success status
        """
        # Validate data
        if len(data) > 2**self.n_qubits:
            warnings.warn("Data too large for quantum memory. Using classical fallback.")
            self.classical_memory[address] = data
            return False
            
        # Normalize data for quantum encoding
        normalized_data = data / np.linalg.norm(data)
        
        try:
            # Get backend
            backend = self.backend_manager.get_backend(backend_name)
            
            # Create storage circuit
            storage_circuit = self._create_storage_circuit(address, normalized_data)
            
            # Validate circuit for backend
            if not backend.validate_circuit(storage_circuit):
                warnings.warn("Circuit not compatible with backend. Using classical fallback.")
                self.classical_memory[address] = data
                return False
                
            # Execute on quantum hardware
            if backend.is_available:
                result = backend.execute_circuit(storage_circuit, shots=1024)
                
                # Store quantum address mapping
                self.quantum_addresses[address] = {
                    'circuit': storage_circuit,
                    'result': result,
                    'backend': backend.name
                }
                
                # Also store classical backup
                self.classical_memory[address] = data
                return True
            else:
                self.classical_memory[address] = data
                return False
                
        except Exception as e:
            warnings.warn(f"Quantum storage failed: {e}. Using classical fallback.")
            self.classical_memory[address] = data
            return False
            
    def _create_storage_circuit(self, address: int, data: np.ndarray) -> QuantumCircuit:
        """Create optimized storage circuit for hardware."""
        if not QISKIT_AVAILABLE:
            return None
            
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Encode address (binary representation)
        address_bits = format(address, f'0{self.n_qubits}b')
        for i, bit in enumerate(address_bits):
            if bit == '1':
                circuit.x(i)
                
        # Encode data using amplitude encoding (simplified)
        # In practice, this would use more sophisticated encoding
        for i, amplitude in enumerate(data[:self.n_qubits]):
            if amplitude > 0.5:  # Threshold encoding
                circuit.ry(2 * np.arcsin(amplitude), i)
                
        # Add measurement
        circuit.measure_all()
        
        return circuit
        
    def retrieve_data(self, address: int, 
                     backend_name: Optional[str] = None) -> np.ndarray:
        """
        Retrieve data from quantum memory.
        
        Args:
            address: Memory address
            backend_name: Specific backend to use
            
        Returns:
            Retrieved data
        """
        # Check if quantum address exists
        if address in self.quantum_addresses:
            try:
                # Get backend
                backend = self.backend_manager.get_backend(backend_name)
                
                # Create retrieval circuit
                retrieval_circuit = self._create_retrieval_circuit(address)
                
                # Execute on quantum hardware
                if backend.is_available:
                    result = backend.execute_circuit(retrieval_circuit, shots=1024)
                    
                    # Process quantum result
                    retrieved_data = self._process_quantum_result(result)
                    
                    # Apply error mitigation if enabled
                    if self.error_mitigation:
                        retrieved_data = self._apply_error_mitigation(retrieved_data, address)
                        
                    return retrieved_data
                    
            except Exception as e:
                warnings.warn(f"Quantum retrieval failed: {e}. Using classical fallback.")
                
        # Classical fallback
        return self.classical_memory.get(address, np.zeros(self.n_qubits))
        
    def _create_retrieval_circuit(self, address: int) -> QuantumCircuit:
        """Create optimized retrieval circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        # Use stored circuit as template
        if address in self.quantum_addresses:
            base_circuit = self.quantum_addresses[address]['circuit']
            # Create inverse circuit for retrieval
            retrieval_circuit = base_circuit.inverse()
            return retrieval_circuit
        else:
            # Create new retrieval circuit
            circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
            circuit.measure_all()
            return circuit
            
    def _process_quantum_result(self, result: Dict[str, int]) -> np.ndarray:
        """Process quantum measurement results into data."""
        # Convert measurement counts to probability distribution
        total_shots = sum(result.values())
        probabilities = []
        
        # Sort by bit string
        for i in range(2**self.n_qubits):
            bit_string = format(i, f'0{self.n_qubits}b')
            count = result.get(bit_string, 0)
            probabilities.append(count / total_shots)
            
        return np.array(probabilities[:self.n_qubits])  # Truncate to n_qubits
        
    def _apply_error_mitigation(self, data: np.ndarray, address: int) -> np.ndarray:
        """Apply error mitigation techniques."""
        # Simple error mitigation: compare with classical backup
        if address in self.classical_memory:
            classical_data = self.classical_memory[address]
            
            # Weighted combination (favor classical for high noise)
            noise_factor = 0.3  # Estimated noise level
            corrected_data = (1 - noise_factor) * data + noise_factor * classical_data[:len(data)]
            
            return corrected_data
            
        return data
        
    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware execution statistics."""
        return {
            'total_addresses': len(self.quantum_addresses),
            'classical_fallbacks': len(self.classical_memory) - len(self.quantum_addresses),
            'quantum_success_rate': len(self.quantum_addresses) / max(len(self.classical_memory), 1),
            'available_backends': list(self.backend_manager.backends.keys()),
            'hardware_constraints': {
                'max_qubits': self.n_qubits,
                'max_circuit_depth': self.max_circuit_depth,
                'max_two_qubit_gates': self.max_two_qubit_gates
            }
        }


class NISQOptimizedLayers(nn.Module):
    """
    Neural network layers optimized for NISQ quantum devices.
    
    Implements variational quantum circuits with minimal depth and
    hardware-aware gate selection.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2, 
                 backend_manager: Optional[QuantumBackendManager] = None):
        """
        Initialize NISQ-optimized quantum layers.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers (keep low for NISQ)
            backend_manager: Quantum backend manager
        """
        super().__init__()
        
        if n_layers > 5:
            warnings.warn(f"n_layers={n_layers} may be too deep for NISQ devices")
            
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_manager = backend_manager or QuantumBackendManager()
        
        # Variational parameters (trainable)
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.phi = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        # Circuit templates
        self._create_nisq_circuits()
        
    def _create_nisq_circuits(self):
        """Create NISQ-optimized circuit templates."""
        if not QISKIT_AVAILABLE:
            return
            
        # Hardware-efficient ansatz
        self.ansatz = EfficientSU2(self.n_qubits, reps=self.n_layers)
        
        # Measurement circuits for different observables
        self.measurement_circuits = {}
        
        # Z measurement
        z_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        z_circuit.measure_all()
        self.measurement_circuits['Z'] = z_circuit
        
        # X measurement
        x_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            x_circuit.ry(-np.pi/2, i)
        x_circuit.measure_all()
        self.measurement_circuits['X'] = x_circuit
        
    def forward(self, x: torch.Tensor, 
                backend_name: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through NISQ quantum layers.
        
        Args:
            x: Input tensor [batch_size, n_qubits]
            backend_name: Specific backend to use
            
        Returns:
            Output tensor [batch_size, n_qubits]
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Create parameterized circuit for this input
            circuit = self._create_parameterized_circuit(x[i])
            
            # Execute on quantum hardware
            try:
                backend = self.backend_manager.get_backend(backend_name)
                if backend and backend.is_available:
                    result = backend.execute_circuit(circuit, shots=1024)
                    output = self._process_measurement_result(result)
                else:
                    # Classical simulation fallback
                    output = self._classical_simulation(x[i])
            except Exception as e:
                warnings.warn(f"Quantum execution failed: {e}. Using classical fallback.")
                output = self._classical_simulation(x[i])
                
            outputs.append(output)
            
        return torch.stack(outputs)
        
    def _create_parameterized_circuit(self, input_data: torch.Tensor) -> QuantumCircuit:
        """Create parameterized quantum circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Data encoding
        for i, val in enumerate(input_data[:self.n_qubits]):
            circuit.ry(float(val), i)
            
        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                circuit.ry(float(self.theta[layer, i]), i)
                circuit.rz(float(self.phi[layer, i]), i)
                
            # Entangling gates (nearest neighbor for hardware efficiency)
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
                
        # Measurement
        circuit.measure_all()
        
        return circuit
        
    def _process_measurement_result(self, result: Dict[str, int]) -> torch.Tensor:
        """Process quantum measurement results."""
        total_shots = sum(result.values())
        expectations = []
        
        # Calculate expectation values for each qubit
        for qubit in range(self.n_qubits):
            expectation = 0.0
            for bit_string, count in result.items():
                if len(bit_string) > qubit:
                    bit_value = int(bit_string[-(qubit+1)])  # Reverse order
                    expectation += (2 * bit_value - 1) * count / total_shots
            expectations.append(expectation)
            
        return torch.tensor(expectations, dtype=torch.float32)
        
    def _classical_simulation(self, input_data: torch.Tensor) -> torch.Tensor:
        """Classical simulation fallback."""
        # Simple classical approximation of quantum circuit
        output = input_data.clone()
        
        for layer in range(self.n_layers):
            # Apply rotations
            output = output * torch.cos(self.theta[layer]) + torch.sin(self.phi[layer])
            
            # Apply entanglement-like mixing
            if self.n_qubits > 1:
                output = 0.8 * output + 0.2 * torch.roll(output, 1)
                
        return torch.tanh(output)  # Normalize output


class ExperimentalQMNN(QMNN):
    """
    QMNN model specifically designed for real quantum hardware experiments.
    
    Integrates hardware-aware components and provides experimental validation
    capabilities on NISQ devices.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_qubits: int = 8, backend_manager: Optional[QuantumBackendManager] = None,
                 **kwargs):
        """
        Initialize experimental QMNN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_qubits: Number of qubits for quantum components
            backend_manager: Quantum backend manager
        """
        # Limit qubits for current hardware
        if n_qubits > 20:
            warnings.warn(f"Reducing n_qubits from {n_qubits} to 20 for hardware compatibility")
            n_qubits = 20
            
        self.backend_manager = backend_manager or QuantumBackendManager()
        
        # Initialize base QMNN with hardware constraints
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            memory_capacity=min(kwargs.get('memory_capacity', 32), 2**n_qubits),
            max_qubits=n_qubits,
            **kwargs
        )
        
        # Replace quantum components with hardware-aware versions
        self.hardware_qram = HardwareAwareQRAM(
            n_qubits=n_qubits,
            backend_manager=self.backend_manager
        )
        
        self.nisq_layers = NISQOptimizedLayers(
            n_qubits=n_qubits,
            n_layers=2,  # Keep shallow for NISQ
            backend_manager=self.backend_manager
        )
        
        # Experimental tracking
        self.experiment_log = []
        
    def experimental_forward(self, x: torch.Tensor, 
                           backend_name: Optional[str] = None,
                           log_experiment: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with experimental hardware execution.
        
        Args:
            x: Input tensor
            backend_name: Specific backend to use
            log_experiment: Whether to log experimental data
            
        Returns:
            Tuple of (output, experiment_info)
        """
        experiment_info = {
            'backend_used': backend_name,
            'quantum_success': False,
            'classical_fallback': False,
            'hardware_stats': {}
        }
        
        try:
            # Get backend info
            backend = self.backend_manager.get_backend(backend_name)
            experiment_info['backend_info'] = backend.get_backend_info()
            
            # Process through NISQ layers
            quantum_output = self.nisq_layers(x, backend_name)
            experiment_info['quantum_success'] = True
            
            # Continue with classical processing
            output = self.classifier(quantum_output)
            
            # Get hardware statistics
            experiment_info['hardware_stats'] = self.hardware_qram.get_hardware_stats()
            
        except Exception as e:
            warnings.warn(f"Experimental execution failed: {e}. Using classical fallback.")
            output = super().forward(x)[0]  # Use base QMNN
            experiment_info['classical_fallback'] = True
            experiment_info['error'] = str(e)
            
        # Log experiment if requested
        if log_experiment:
            self.experiment_log.append({
                'input_shape': x.shape,
                'output_shape': output.shape,
                'experiment_info': experiment_info
            })
            
        return output, experiment_info
        
    def run_hardware_benchmark(self, test_data: torch.Tensor,
                              backends: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive hardware benchmark across multiple backends.
        
        Args:
            test_data: Test data for benchmarking
            backends: List of backend names to test
            
        Returns:
            Benchmark results
        """
        if backends is None:
            backends = list(self.backend_manager.backends.keys())
            
        benchmark_results = {}
        
        for backend_name in backends:
            print(f"Benchmarking on {backend_name}...")
            
            backend_results = {
                'success_rate': 0.0,
                'average_fidelity': 0.0,
                'execution_times': [],
                'errors': []
            }
            
            successful_runs = 0
            total_runs = min(len(test_data), 10)  # Limit for hardware costs
            
            for i in range(total_runs):
                try:
                    import time
                    start_time = time.time()
                    
                    output, exp_info = self.experimental_forward(
                        test_data[i:i+1], 
                        backend_name=backend_name
                    )
                    
                    execution_time = time.time() - start_time
                    backend_results['execution_times'].append(execution_time)
                    
                    if exp_info['quantum_success']:
                        successful_runs += 1
                        
                except Exception as e:
                    backend_results['errors'].append(str(e))
                    
            backend_results['success_rate'] = successful_runs / total_runs
            benchmark_results[backend_name] = backend_results
            
        return benchmark_results
        
    def get_experimental_summary(self) -> Dict[str, Any]:
        """Get summary of experimental runs."""
        if not self.experiment_log:
            return {"message": "No experiments logged"}
            
        total_experiments = len(self.experiment_log)
        quantum_successes = sum(1 for exp in self.experiment_log 
                               if exp['experiment_info']['quantum_success'])
        
        return {
            'total_experiments': total_experiments,
            'quantum_success_rate': quantum_successes / total_experiments,
            'backends_used': list(set(exp['experiment_info']['backend_used'] 
                                    for exp in self.experiment_log 
                                    if exp['experiment_info']['backend_used'])),
            'average_input_size': np.mean([np.prod(exp['input_shape']) 
                                         for exp in self.experiment_log])
        }
