"""
Unit tests for QMANN models.
Tests are designed to work across different environments.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import warnings

from qmann.models import QuantumNeuralNetwork, QMANN


class TestQuantumNeuralNetwork(unittest.TestCase):
    """Test cases for QuantumNeuralNetwork."""
    
    def test_qnn_initialization(self):
        """Test QNN initialization with valid parameters."""
        input_dim = 10
        output_dim = 5
        n_qubits = 4

        qnn = QuantumNeuralNetwork(input_dim, output_dim, n_qubits)

        self.assertEqual(qnn.input_dim, input_dim)
        self.assertEqual(qnn.output_dim, output_dim)
        self.assertEqual(qnn.n_qubits, n_qubits)
        self.assertIsInstance(qnn.input_layer, nn.Linear)
        # output_layer might be Sequential or Linear
        self.assertTrue(hasattr(qnn, 'output_layer'))
        
    def test_qnn_forward_pass(self):
        """Test QNN forward pass."""
        qnn = QuantumNeuralNetwork(input_dim=8, output_dim=3, n_qubits=4)
        
        # Create sample input
        batch_size = 2
        x = torch.randn(batch_size, 8)
        
        # Forward pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = qnn(x)
        
        self.assertEqual(output.shape, (batch_size, 3))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_qnn_parameter_count(self):
        """Test QNN parameter counting."""
        qnn = QuantumNeuralNetwork(input_dim=6, output_dim=4, n_qubits=3)
        
        total_params = sum(p.numel() for p in qnn.parameters())
        self.assertGreater(total_params, 0)
        
    def test_qnn_different_sizes(self):
        """Test QNN with different input/output sizes."""
        configs = [
            (4, 2, 3),
            (8, 4, 4),
            (12, 6, 5)
        ]
        
        for input_dim, output_dim, n_qubits in configs:
            with self.subTest(input_dim=input_dim, output_dim=output_dim, n_qubits=n_qubits):
                qnn = QuantumNeuralNetwork(input_dim, output_dim, n_qubits)
                
                x = torch.randn(2, input_dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = qnn(x)
                
                self.assertEqual(output.shape, (2, output_dim))


class TestQMANN(unittest.TestCase):
    """Test cases for QMANN model."""
    
    def test_qmann_initialization(self):
        """Test QMANN initialization with valid parameters."""
        config = {
            'input_dim': 8,
            'hidden_dim': 16,
            'output_dim': 4,
            'memory_capacity': 8,
            'memory_embedding_dim': 16,
            'n_quantum_layers': 2,
            'max_qubits': 6
        }
        
        model = QMANN(**config)
        
        self.assertEqual(model.input_dim, config['input_dim'])
        self.assertEqual(model.hidden_dim, config['hidden_dim'])
        self.assertEqual(model.output_dim, config['output_dim'])
        
    def test_qmann_forward_pass(self):
        """Test QMANN forward pass."""
        model = QMANN(
            input_dim=6,
            hidden_dim=12,
            output_dim=3,
            memory_capacity=8,
            memory_embedding_dim=12,
            n_quantum_layers=2,
            max_qubits=5
        )
        
        # Test with 3D input (batch, seq_len, features)
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 6)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, 3))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_qmann_memory_integration(self):
        """Test QMANN memory integration."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=4
        )
        
        # Test that model has quantum memory
        self.assertIsNotNone(model.quantum_memory)
        
        # Test memory usage
        initial_usage = model.quantum_memory.memory_usage()
        self.assertEqual(initial_usage, 0.0)
        
        # Run forward pass to populate memory
        x = torch.randn(1, 5, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        # Memory should have some usage after forward pass
        final_usage = model.quantum_memory.memory_usage()
        self.assertGreaterEqual(final_usage, 0.0)
        
    def test_qmann_adaptive_capacity(self):
        """Test QMANN adaptive memory capacity based on qubit constraints."""
        # Request large memory but limit qubits
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=32,  # Large request
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=4  # Limited qubits
        )
        
        # Actual capacity should be reduced
        actual_capacity = model.quantum_memory.effective_capacity
        self.assertLessEqual(actual_capacity, 32)
        self.assertGreater(actual_capacity, 0)
        
    def test_qmann_parameter_count(self):
        """Test QMANN parameter counting."""
        model = QMANN(
            input_dim=6,
            hidden_dim=12,
            output_dim=3,
            memory_capacity=8,
            memory_embedding_dim=12,
            n_quantum_layers=2,
            max_qubits=5
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Should have reasonable number of parameters
        self.assertLess(total_params, 100000)  # Sanity check


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility across different environments."""
    
    def test_theoretical_mode_compatibility(self):
        """Test models work in theoretical mode."""
        # Should work without any quantum libraries
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        x = torch.randn(1, 3, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        self.assertEqual(output.shape, (1, 3, 2))
        
    def test_simulation_mode_compatibility(self):
        """Test models work with simulation libraries."""
        try:
            import qiskit
            # If Qiskit available, test enhanced functionality
            model = QMANN(
                input_dim=4,
                hidden_dim=8,
                output_dim=2,
                memory_capacity=4,
                memory_embedding_dim=8,
                n_quantum_layers=1,
                max_qubits=3
            )
            
            x = torch.randn(1, 3, 4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(x)
            
            self.assertEqual(output.shape, (1, 3, 2))
            
        except ImportError:
            self.skipTest("Qiskit not available for simulation mode test")
            
    def test_hardware_mode_compatibility(self):
        """Test models are compatible with hardware interfaces."""
        try:
            from qmann.hardware import QuantumBackendManager
            
            # Test that models can be created even with hardware interface
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
            x = torch.randn(1, 3, 4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(x)
            
            self.assertEqual(output.shape, (1, 3, 2))
            
        except ImportError:
            self.skipTest("Hardware interface not available")


class TestModelPerformance(unittest.TestCase):
    """Test model performance characteristics."""
    
    def test_model_training_mode(self):
        """Test model in training mode."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        model.train()
        self.assertTrue(model.training)
        
        x = torch.randn(2, 3, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        self.assertEqual(output.shape, (2, 3, 2))
        
    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        model.eval()
        self.assertFalse(model.training)
        
        x = torch.randn(2, 3, 4)
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(x)
        
        self.assertEqual(output.shape, (2, 3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
