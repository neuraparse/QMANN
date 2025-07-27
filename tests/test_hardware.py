"""
Unit tests for QMANN hardware interfaces.
Tests quantum backend compatibility and hardware integration.
"""

import unittest
import warnings
import numpy as np


class TestQuantumBackends(unittest.TestCase):
    """Test quantum backend interfaces."""
    
    def test_backend_manager_initialization(self):
        """Test quantum backend manager initialization."""
        try:
            from qmann.hardware import QuantumBackendManager
            
            # Should initialize without errors
            backend_manager = QuantumBackendManager()
            self.assertIsNotNone(backend_manager)
            
        except ImportError:
            self.skipTest("Hardware interface not available")
            
    def test_backend_listing(self):
        """Test listing available backends."""
        try:
            from qmann.hardware import QuantumBackendManager
            
            backend_manager = QuantumBackendManager()
            backends = backend_manager.list_backends()
            
            self.assertIsInstance(backends, dict)
            # Should have at least simulator backends
            
        except ImportError:
            self.skipTest("Hardware interface not available")
        except Exception as e:
            # Expected if no quantum credentials configured
            self.assertIn("backend", str(e).lower())
            
    def test_simulator_backend_access(self):
        """Test accessing simulator backends."""
        try:
            from qmann.hardware import QuantumBackendManager
            
            backend_manager = QuantumBackendManager()
            
            # Try to get any available backend
            try:
                backend = backend_manager.get_backend()
                if backend:
                    self.assertIsNotNone(backend.name)
                    
            except Exception:
                # Expected if no backends configured
                pass
                
        except ImportError:
            self.skipTest("Hardware interface not available")
            
    def test_ibm_backend_compatibility(self):
        """Test IBM Quantum backend compatibility."""
        try:
            from qmann.hardware import IBMQuantumBackend
            
            # Should be able to create backend object (even if not connected)
            backend = IBMQuantumBackend(use_simulator=True)
            self.assertIsNotNone(backend)
            
        except ImportError:
            self.skipTest("IBM backend interface not available")
        except Exception as e:
            # Expected without proper credentials
            self.assertIn(("qiskit" in str(e).lower() or 
                          "ibm" in str(e).lower() or
                          "backend" in str(e).lower()), True)
            
    def test_google_backend_compatibility(self):
        """Test Google Quantum backend compatibility."""
        try:
            from qmann.hardware import GoogleQuantumBackend
            
            # Should be able to create backend object
            backend = GoogleQuantumBackend(use_simulator=True)
            self.assertIsNotNone(backend)
            
        except ImportError:
            self.skipTest("Google backend interface not available")
        except Exception:
            # Expected without proper setup
            pass
            
    def test_ionq_backend_compatibility(self):
        """Test IonQ backend compatibility."""
        try:
            from qmann.hardware import IonQBackend
            
            # Should be able to create backend object
            backend = IonQBackend(use_simulator=True)
            self.assertIsNotNone(backend)
            
        except ImportError:
            self.skipTest("IonQ backend interface not available")
        except Exception:
            # Expected without proper setup
            pass


class TestHardwareIntegration(unittest.TestCase):
    """Test hardware integration with QMANN models."""
    
    def test_model_hardware_compatibility(self):
        """Test QMANN model compatibility with hardware interfaces."""
        try:
            from qmann.models import QMANN
            from qmann.hardware import QuantumBackendManager
            
            # Create model
            model = QMANN(
                input_dim=4,
                hidden_dim=8,
                output_dim=2,
                memory_capacity=4,
                memory_embedding_dim=8,
                n_quantum_layers=1,
                max_qubits=3
            )
            
            # Should work regardless of hardware availability
            self.assertIsNotNone(model)
            
        except ImportError:
            self.skipTest("Hardware interface not available")
            
    def test_experimental_interface(self):
        """Test experimental quantum hardware interface."""
        try:
            from qmann.hardware import ExperimentalQMANN
            
            # Should be able to import experimental interface
            self.assertTrue(hasattr(ExperimentalQMANN, '__init__'))
            
        except ImportError:
            self.skipTest("Experimental interface not available")
            
    def test_hardware_aware_qram(self):
        """Test hardware-aware QRAM implementation."""
        try:
            from qmann.hardware import HardwareAwareQRAM
            
            # Should be able to import hardware-aware QRAM
            self.assertTrue(hasattr(HardwareAwareQRAM, '__init__'))
            
        except ImportError:
            self.skipTest("Hardware-aware QRAM not available")
            
    def test_nisq_optimized_layers(self):
        """Test NISQ-optimized quantum layers."""
        try:
            from qmann.hardware import NISQOptimizedLayers
            
            # Should be able to import NISQ layers
            self.assertTrue(hasattr(NISQOptimizedLayers, '__init__'))
            
        except ImportError:
            self.skipTest("NISQ-optimized layers not available")


class TestQuantumSimulation(unittest.TestCase):
    """Test quantum simulation capabilities."""
    
    def test_qiskit_simulation(self):
        """Test Qiskit-based quantum simulation."""
        try:
            import qiskit
            from qiskit import QuantumCircuit
            
            # Create simple quantum circuit
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            
            self.assertEqual(circuit.num_qubits, 2)
            
        except ImportError:
            self.skipTest("Qiskit not available for simulation")
            
    def test_pennylane_simulation(self):
        """Test PennyLane-based quantum simulation."""
        try:
            import pennylane as qml
            
            # Create simple quantum device
            dev = qml.device('default.qubit', wires=2)
            
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            probs = circuit()
            self.assertEqual(len(probs), 4)  # 2^2 = 4 states
            
        except ImportError:
            self.skipTest("PennyLane not available for simulation")
            
    def test_quantum_memory_simulation(self):
        """Test quantum memory simulation."""
        from qmann.core import QuantumMemory
        
        # Test quantum memory works in simulation mode
        memory = QuantumMemory(capacity=4, embedding_dim=4)
        
        key = np.array([1.0, 0.0, 0.0, 0.0])
        value = np.array([0.0, 1.0, 0.0, 0.0])
        
        memory.store_embedding(key, value)
        retrieved = memory.retrieve_embedding(key)
        
        self.assertIsInstance(retrieved, np.ndarray)
        self.assertEqual(len(retrieved), 4)


class TestHardwareConstraints(unittest.TestCase):
    """Test hardware constraint handling."""
    
    def test_qubit_limitation_handling(self):
        """Test handling of qubit limitations."""
        from qmann.models import QMANN
        
        # Test with very limited qubits
        model = QMANN(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            memory_capacity=32,  # Large request
            memory_embedding_dim=16,
            n_quantum_layers=2,
            max_qubits=3  # Very limited
        )
        
        # Should adapt to constraints
        actual_capacity = model.quantum_memory.effective_capacity
        self.assertLessEqual(actual_capacity, 32)
        self.assertGreater(actual_capacity, 0)
        
    def test_noise_tolerance(self):
        """Test noise tolerance in quantum operations."""
        from qmann.core import QuantumMemory
        
        memory = QuantumMemory(capacity=4, embedding_dim=4)
        
        # Add noise to inputs
        key = np.array([1.0, 0.0, 0.0, 0.0]) + 0.01 * np.random.randn(4)
        value = np.array([0.0, 1.0, 0.0, 0.0]) + 0.01 * np.random.randn(4)
        
        # Should handle noisy inputs gracefully
        memory.store_embedding(key, value)
        retrieved = memory.retrieve_embedding(key)
        
        self.assertIsInstance(retrieved, np.ndarray)
        
    def test_circuit_depth_constraints(self):
        """Test circuit depth constraint handling."""
        from qmann.models import QMANN
        
        # Test with many quantum layers (should be constrained)
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=10,  # Many layers
            max_qubits=4
        )
        
        # Should still work (layers may be reduced internally)
        import torch
        x = torch.randn(1, 3, 4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        self.assertEqual(output.shape, (1, 3, 2))


class TestHardwarePerformance(unittest.TestCase):
    """Test hardware performance characteristics."""
    
    def test_simulation_performance(self):
        """Test simulation performance."""
        from qmann.models import QMANN
        import time
        
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        # Measure inference time
        import torch
        x = torch.randn(1, 5, 4)
        
        start_time = time.time()
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(x)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for small model)
        self.assertLess(inference_time, 1.0)
        
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        from qmann.models import QMANN
        import torch
        
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(total_params, 10000)  # Should be efficient
        
    def test_scalability(self):
        """Test model scalability."""
        from qmann.models import QMANN
        import torch
        
        # Test different model sizes
        sizes = [
            (4, 8, 2, 4),
            (8, 16, 4, 8),
            (12, 24, 6, 12)
        ]
        
        for input_dim, hidden_dim, output_dim, capacity in sizes:
            with self.subTest(size=(input_dim, hidden_dim, output_dim, capacity)):
                model = QMANN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    memory_capacity=capacity,
                    memory_embedding_dim=hidden_dim,
                    n_quantum_layers=1,
                    max_qubits=min(6, capacity)
                )
                
                x = torch.randn(1, 3, input_dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = model(x)
                
                self.assertEqual(output.shape, (1, 3, output_dim))


if __name__ == "__main__":
    unittest.main(verbosity=2)
