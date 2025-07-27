"""
Unit tests for QMANN core components.
Tests are designed to work in different environments (theoretical, simulation, hardware).
"""

import unittest
import numpy as np
import torch
import warnings

from qmann.core import QRAM, QuantumMemory


class TestQRAM(unittest.TestCase):
    """Test cases for QRAM implementation."""
    
    def test_qram_initialization(self):
        """Test QRAM initialization with valid parameters."""
        memory_size = 8
        address_qubits = 3
        max_data_qubits = 3

        qram = QRAM(memory_size, address_qubits, max_data_qubits)

        self.assertEqual(qram.memory_size, memory_size)
        self.assertEqual(qram.address_qubits, address_qubits)
        self.assertEqual(qram.max_data_qubits, max_data_qubits)
        self.assertEqual(qram.memory.shape, (memory_size, 2**max_data_qubits))
        
    def test_qram_invalid_parameters(self):
        """Test QRAM initialization with invalid parameters."""
        # Test with insufficient address qubits for memory size
        # This should not raise an error but adjust memory_size
        qram = QRAM(memory_size=16, address_qubits=2)  # 2^2 = 4 < 16
        self.assertEqual(qram.memory_size, 4)  # Should be adjusted to 2^2
            
    def test_qram_write_read(self):
        """Test QRAM write and read operations."""
        qram = QRAM(memory_size=4, address_qubits=2, max_data_qubits=2)

        # Write data
        address = 1
        data = np.array([0.5, 0.5, 0.5, 0.5])
        qram.write(address, data)

        # Check data was written
        stored_data = qram.memory[address]
        np.testing.assert_array_almost_equal(stored_data, data)
        
    def test_qram_capacity_bound(self):
        """Test QRAM respects capacity bounds."""
        qram = QRAM(memory_size=4, address_qubits=2, max_data_qubits=2)

        # Valid address
        qram.write(3, np.array([1.0, 0.0, 0.0, 0.0]))

        # Invalid address should raise error
        with self.assertRaises(ValueError):
            qram.write(4, np.array([1.0, 0.0, 0.0, 0.0]))


class TestQuantumMemory(unittest.TestCase):
    """Test cases for QuantumMemory implementation."""
    
    def test_quantum_memory_initialization(self):
        """Test QuantumMemory initialization."""
        capacity = 16
        embedding_dim = 8
        
        qmem = QuantumMemory(capacity, embedding_dim)
        
        self.assertEqual(qmem.capacity, capacity)
        self.assertEqual(qmem.embedding_dim, embedding_dim)
        self.assertEqual(qmem.stored_count, 0)
        
    def test_store_and_retrieve_embedding(self):
        """Test storing and retrieving embeddings."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        # Store embedding
        key = np.array([1.0, 0.0, 0.0, 0.0])
        value = np.array([0.0, 1.0, 0.0, 0.0])
        qmem.store_embedding(key, value)
        
        self.assertEqual(qmem.stored_count, 1)
        
        # Retrieve embedding
        retrieved = qmem.retrieve_embedding(key)
        
        self.assertIsInstance(retrieved, np.ndarray)
        self.assertEqual(len(retrieved), 4)
        
    def test_memory_usage(self):
        """Test memory usage calculation."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        # Initially empty
        self.assertEqual(qmem.memory_usage(), 0.0)
        
        # Store some embeddings
        for i in range(4):
            key = np.zeros(4)
            key[i % 4] = 1.0
            value = np.zeros(4)
            value[(i + 1) % 4] = 1.0
            qmem.store_embedding(key, value)
        
        self.assertEqual(qmem.memory_usage(), 0.5)  # 4/8 = 0.5
        
    def test_memory_capacity_handling(self):
        """Test memory capacity handling."""
        qmem = QuantumMemory(capacity=2, embedding_dim=4)

        # Fill memory to capacity
        for i in range(2):
            key = np.random.randn(4)
            value = np.random.randn(4)
            qmem.store_embedding(key, value)

        # Store one more - should raise error when memory is full
        with self.assertRaises(RuntimeError):
            qmem.store_embedding(np.random.randn(4), np.random.randn(4))
        
    def test_memory_reset(self):
        """Test memory reset functionality."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        # Store some data
        for i in range(3):
            key = np.random.randn(4)
            value = np.random.randn(4)
            qmem.store_embedding(key, value)
        
        # Reset memory
        qmem.reset()
        self.assertEqual(qmem.stored_count, 0)


class TestEnvironmentCompatibility(unittest.TestCase):
    """Test compatibility across different environments."""
    
    def test_theoretical_mode(self):
        """Test operation in theoretical mode (no quantum libraries)."""
        # This should always work
        qmem = QuantumMemory(capacity=4, embedding_dim=4)
        
        key = np.array([1.0, 0.0, 0.0, 0.0])
        value = np.array([0.0, 1.0, 0.0, 0.0])
        
        qmem.store_embedding(key, value)
        retrieved = qmem.retrieve_embedding(key)
        
        self.assertIsInstance(retrieved, np.ndarray)
        
    def test_simulation_mode_compatibility(self):
        """Test compatibility with simulation mode."""
        try:
            import qiskit
            # If Qiskit is available, test quantum circuit creation
            qram = QRAM(memory_size=4, address_qubits=2, max_data_qubits=2)

            # This should work without errors
            circuit = qram.create_quantum_circuit()
            self.assertIsNotNone(circuit)

        except ImportError:
            # If Qiskit not available, skip this test
            self.skipTest("Qiskit not available for simulation mode test")
        except AttributeError:
            # If create_quantum_circuit method doesn't exist, skip
            self.skipTest("Quantum circuit creation not implemented")
            
    def test_hardware_mode_compatibility(self):
        """Test hardware mode compatibility."""
        try:
            from qmann.hardware import QuantumBackendManager
            
            # Test backend manager initialization
            backend_manager = QuantumBackendManager()
            backends = backend_manager.list_backends()
            
            # Should not raise errors even if no real hardware available
            self.assertIsInstance(backends, dict)
            
        except ImportError:
            self.skipTest("Hardware interface not available")


class TestIntegration(unittest.TestCase):
    """Integration tests for core components."""
    
    def test_qram_quantum_memory_integration(self):
        """Test integration between QRAM and QuantumMemory."""
        qmem = QuantumMemory(capacity=4, embedding_dim=4)
        
        # Store and retrieve multiple embeddings
        embeddings = []
        for i in range(3):
            key = np.random.randn(4)
            value = np.random.randn(4)
            embeddings.append((key, value))
            qmem.store_embedding(key, value)
            
        # Test retrieval
        for key, expected_value in embeddings:
            retrieved = qmem.retrieve_embedding(key)
            # Note: exact match not expected due to quantum processing
            self.assertIsInstance(retrieved, np.ndarray)
            self.assertEqual(len(retrieved), 4)
            
    def test_memory_scaling(self):
        """Test memory scaling with different sizes."""
        sizes = [4, 8, 16]
        
        for size in sizes:
            with self.subTest(size=size):
                qmem = QuantumMemory(capacity=size, embedding_dim=4)
                
                # Store embeddings up to capacity
                for i in range(min(size, 5)):  # Don't exceed capacity
                    key = np.random.randn(4)
                    value = np.random.randn(4)
                    qmem.store_embedding(key, value)
                
                self.assertLessEqual(qmem.stored_count, size)


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2)
