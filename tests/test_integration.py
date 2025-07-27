"""
Integration tests for QMANN system.
Tests end-to-end functionality across different modes and environments.
"""

import unittest
import torch
import numpy as np
import warnings
import tempfile
import os


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def test_complete_training_workflow(self):
        """Test complete training workflow from model creation to evaluation."""
        from qmann.models import QMANN
        from qmann.training import QMANNTrainer
        
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
        
        # Create trainer
        trainer = QMANNTrainer(model=model, learning_rate=0.01, device='cpu')
        
        # Create dummy dataset
        train_data = []
        val_data = []
        
        for _ in range(10):  # Small dataset for testing
            x = torch.randn(2, 3, 4)
            y = torch.randn(2, 3, 2)
            train_data.append((x, y))
            
        for _ in range(5):
            x = torch.randn(2, 3, 4)
            y = torch.randn(2, 3, 2)
            val_data.append((x, y))
        
        # Train for a few epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for epoch in range(3):
                # Training
                train_loss = trainer.train_epoch(train_data)
                
                # Validation
                val_losses = []
                for x, y in val_data:
                    val_loss = trainer.validate_step(x, y)
                    val_losses.append(val_loss)
                
                avg_val_loss = np.mean(val_losses)
                
                self.assertIsInstance(train_loss, float)
                self.assertIsInstance(avg_val_loss, float)
                self.assertGreater(train_loss, 0)
                self.assertGreater(avg_val_loss, 0)
        
    def test_model_persistence_workflow(self):
        """Test model saving and loading workflow."""
        from qmann.models import QMANN
        from qmann.training import QMANNTrainer
        
        # Create and train model
        model1 = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer1 = QMANNTrainer(model=model1, device='cpu')
        
        # Train briefly
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss1 = trainer1.train_step(x, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        try:
            trainer1.save_model(temp_path)
            
            # Create new model and load
            model2 = QMANN(
                input_dim=4,
                hidden_dim=8,
                output_dim=2,
                memory_capacity=4,
                memory_embedding_dim=8,
                n_quantum_layers=1,
                max_qubits=3
            )
            
            trainer2 = QMANNTrainer(model=model2, device='cpu')
            trainer2.load_model(temp_path)
            
            # Test that loaded model produces same output
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output1 = model1(x)
                    output2 = model2(x)
            
            # Outputs should be close (allowing for small numerical differences)
            self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_memory_evolution_workflow(self):
        """Test quantum memory evolution during training."""
        from qmann.models import QMANN
        from qmann.training import QMANNTrainer
        
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=8,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=4
        )
        
        trainer = QMANNTrainer(model=model, device='cpu')
        
        # Track memory usage over time
        memory_usage_history = []
        
        # Initial memory usage
        initial_usage = model.quantum_memory.memory_usage()
        memory_usage_history.append(initial_usage)
        
        # Train with different data
        for i in range(5):
            x = torch.randn(1, 3, 4)
            y = torch.randn(1, 3, 2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train_step(x, y)
            
            usage = model.quantum_memory.memory_usage()
            memory_usage_history.append(usage)
        
        # Memory usage should evolve
        self.assertEqual(len(memory_usage_history), 6)
        
        # All usage values should be valid
        for usage in memory_usage_history:
            self.assertGreaterEqual(usage, 0.0)
            self.assertLessEqual(usage, 1.0)


class TestMultiModeCompatibility(unittest.TestCase):
    """Test compatibility across different operational modes."""
    
    def test_theoretical_to_simulation_transition(self):
        """Test transition from theoretical to simulation mode."""
        from qmann.models import QMANN
        
        # Create model (should work in theoretical mode)
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
        
        # Should work regardless of quantum library availability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
        
        self.assertEqual(output.shape, (1, 3, 2))
        
    def test_simulation_mode_features(self):
        """Test simulation mode specific features."""
        try:
            import qiskit
            
            from qmann.models import QMANN
            from qmann.core import QuantumMemory
            
            # Test enhanced quantum memory with Qiskit
            memory = QuantumMemory(capacity=8, embedding_dim=4)
            
            # Store multiple embeddings
            for i in range(4):
                key = np.zeros(4)
                key[i] = 1.0
                value = np.random.randn(4)
                memory.store_embedding(key, value)
            
            # Test retrieval
            test_key = np.array([1.0, 0.0, 0.0, 0.0])
            retrieved = memory.retrieve_embedding(test_key)
            
            self.assertIsInstance(retrieved, np.ndarray)
            self.assertEqual(len(retrieved), 4)
            
        except ImportError:
            self.skipTest("Qiskit not available for simulation mode test")
            
    def test_hardware_mode_preparation(self):
        """Test hardware mode preparation and compatibility."""
        try:
            from qmann.hardware import QuantumBackendManager
            from qmann.models import QMANN
            
            # Test that models are compatible with hardware interface
            model = QMANN(
                input_dim=4,
                hidden_dim=8,
                output_dim=2,
                memory_capacity=4,
                memory_embedding_dim=8,
                n_quantum_layers=1,
                max_qubits=3
            )
            
            # Should work even if hardware not available
            x = torch.randn(1, 3, 4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(x)
            
            self.assertEqual(output.shape, (1, 3, 2))
            
        except ImportError:
            self.skipTest("Hardware interface not available")


class TestSystemRobustness(unittest.TestCase):
    """Test system robustness and error handling."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        from qmann.models import QMANN
        
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        # Test with wrong input dimensions
        x_wrong = torch.randn(1, 3, 6)  # Wrong last dimension
        
        with self.assertRaises(RuntimeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model(x_wrong)
                
    def test_memory_overflow_handling(self):
        """Test handling of memory overflow conditions."""
        from qmann.core import QuantumMemory
        
        memory = QuantumMemory(capacity=2, embedding_dim=4)
        
        # Fill beyond capacity
        for i in range(5):  # More than capacity
            key = np.random.randn(4)
            value = np.random.randn(4)
            
            # Should not raise error, but handle gracefully
            memory.store_embedding(key, value)
        
        # Memory should still be functional
        usage = memory.memory_usage()
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)
        
    def test_gradient_flow_integrity(self):
        """Test gradient flow integrity through quantum components."""
        from qmann.models import QMANN
        from qmann.training import QMANNTrainer
        
        model = QMANN(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            memory_capacity=4,
            memory_embedding_dim=8,
            n_quantum_layers=1,
            max_qubits=3
        )
        
        trainer = QMANNTrainer(model=model, device='cpu')
        
        x = torch.randn(1, 3, 4, requires_grad=True)
        y = torch.randn(1, 3, 2)
        
        # Forward pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check that input gradients exist
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        
    def test_numerical_stability(self):
        """Test numerical stability of quantum operations."""
        from qmann.core import QuantumMemory
        
        memory = QuantumMemory(capacity=4, embedding_dim=4)
        
        # Test with extreme values
        extreme_key = np.array([1e6, -1e6, 1e-6, -1e-6])
        extreme_value = np.array([1e3, -1e3, 1e-3, -1e-3])
        
        # Should handle extreme values gracefully
        memory.store_embedding(extreme_key, extreme_value)
        retrieved = memory.retrieve_embedding(extreme_key)
        
        # Retrieved values should be finite
        self.assertTrue(np.all(np.isfinite(retrieved)))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks across different configurations."""
    
    def test_inference_speed_benchmark(self):
        """Test inference speed across different model sizes."""
        from qmann.models import QMANN
        import time
        
        configs = [
            (4, 8, 2, 4),    # Small
            (8, 16, 4, 8),   # Medium
            (12, 24, 6, 12)  # Large
        ]
        
        for input_dim, hidden_dim, output_dim, capacity in configs:
            with self.subTest(config=(input_dim, hidden_dim, output_dim, capacity)):
                model = QMANN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    memory_capacity=capacity,
                    memory_embedding_dim=hidden_dim,
                    n_quantum_layers=1,
                    max_qubits=min(6, capacity)
                )
                
                x = torch.randn(1, 10, input_dim)
                
                # Warm up
                with torch.no_grad():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = model(x)
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        output = model(x)
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Should complete in reasonable time
                self.assertLess(inference_time, 2.0)  # 2 seconds max
                self.assertEqual(output.shape, (1, 10, output_dim))
                
    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency of different configurations."""
        from qmann.models import QMANN
        
        configs = [
            (4, 8, 2, 4),
            (8, 16, 4, 8),
            (12, 24, 6, 12)
        ]
        
        for input_dim, hidden_dim, output_dim, capacity in configs:
            with self.subTest(config=(input_dim, hidden_dim, output_dim, capacity)):
                model = QMANN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    memory_capacity=capacity,
                    memory_embedding_dim=hidden_dim,
                    n_quantum_layers=1,
                    max_qubits=min(6, capacity)
                )
                
                # Check parameter efficiency
                total_params = sum(p.numel() for p in model.parameters())
                
                # Should scale reasonably with model size
                expected_max_params = input_dim * hidden_dim * 50  # More generous estimate
                self.assertLess(total_params, expected_max_params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
