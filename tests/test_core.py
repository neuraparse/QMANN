"""
Unit tests for QMNN core components.
"""

import pytest
import numpy as np
import torch
from qiskit.quantum_info import Statevector

from qmnn.core import QRAM, QuantumMemory
from qmnn.utils import quantum_state_to_classical, classical_to_quantum_state


class TestQRAM:
    """Test cases for QRAM implementation."""
    
    def test_qram_initialization(self):
        """Test QRAM initialization with valid parameters."""
        memory_size = 8
        address_qubits = 3
        
        qram = QRAM(memory_size, address_qubits)
        
        assert qram.memory_size == memory_size
        assert qram.address_qubits == address_qubits
        assert qram.data_qubits == 3  # ceil(log2(8))
        assert qram.memory.shape == (memory_size, 2**qram.data_qubits)
        
    def test_qram_invalid_parameters(self):
        """Test QRAM initialization with invalid parameters."""
        with pytest.raises(ValueError):
            QRAM(memory_size=16, address_qubits=2)  # 2^2 < 16
            
    def test_qram_write_read(self):
        """Test basic write and read operations."""
        qram = QRAM(memory_size=4, address_qubits=2)
        
        # Write data
        data = np.array([1.0, 0.0, 0.0, 0.0])
        qram.write(address=0, data=data)
        
        # Verify data is stored and normalized
        stored_data = qram.memory[0]
        assert np.allclose(stored_data, data / np.linalg.norm(data))
        
    def test_qram_superposition_read(self):
        """Test reading with superposition addressing."""
        qram = QRAM(memory_size=4, address_qubits=2)
        
        # Store data at different addresses
        qram.write(0, np.array([1.0, 0.0, 0.0, 0.0]))
        qram.write(1, np.array([0.0, 1.0, 0.0, 0.0]))
        
        # Create superposition address state
        address_state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0])
        
        # Read with superposition
        result = qram.read(address_state)
        
        assert isinstance(result, Statevector)
        assert len(result.data) == 2**qram.data_qubits
        
    def test_qram_capacity_bound(self):
        """Test theoretical capacity bound calculation."""
        qram = QRAM(memory_size=8, address_qubits=3)
        capacity = qram.capacity_bound()
        
        expected_capacity = 2**3 * 3  # 2^n * m
        assert capacity == expected_capacity
        
    def test_qram_circuit_creation(self):
        """Test quantum circuit creation."""
        qram = QRAM(memory_size=4, address_qubits=2)
        
        # Store some data
        qram.write(0, np.array([1.0, 0.0, 0.0, 0.0]))
        
        circuit = qram.create_circuit()
        
        assert circuit is not None
        assert circuit.num_qubits == qram.address_qubits + qram.data_qubits


class TestQuantumMemory:
    """Test cases for QuantumMemory implementation."""
    
    def test_quantum_memory_initialization(self):
        """Test QuantumMemory initialization."""
        capacity = 16
        embedding_dim = 8
        
        qmem = QuantumMemory(capacity, embedding_dim)
        
        assert qmem.capacity == capacity
        assert qmem.embedding_dim == embedding_dim
        assert qmem.address_qubits == 4  # ceil(log2(16))
        assert qmem.data_qubits == 3     # ceil(log2(8))
        assert qmem.stored_count == 0
        
    def test_store_embedding(self):
        """Test storing key-value embeddings."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        key = np.array([1.0, 0.0, 0.0, 0.0])
        value = np.array([0.0, 1.0, 0.0, 0.0])
        
        address = qmem.store_embedding(key, value)
        
        assert isinstance(address, int)
        assert 0 <= address < qmem.capacity
        assert qmem.stored_count == 1
        
    def test_retrieve_embedding(self):
        """Test retrieving embeddings."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        # Store some embeddings
        key1 = np.array([1.0, 0.0, 0.0, 0.0])
        value1 = np.array([0.0, 1.0, 0.0, 0.0])
        qmem.store_embedding(key1, value1)
        
        key2 = np.array([0.0, 1.0, 0.0, 0.0])
        value2 = np.array([0.0, 0.0, 1.0, 0.0])
        qmem.store_embedding(key2, value2)
        
        # Retrieve with query
        query = np.array([1.0, 0.0, 0.0, 0.0])
        retrieved = qmem.retrieve_embedding(query)
        
        assert isinstance(retrieved, np.ndarray)
        assert len(retrieved) == qmem.embedding_dim
        
    def test_memory_capacity_exceeded(self):
        """Test behavior when memory capacity is exceeded."""
        qmem = QuantumMemory(capacity=2, embedding_dim=4)
        
        # Fill memory to capacity
        for i in range(2):
            key = np.random.randn(4)
            value = np.random.randn(4)
            qmem.store_embedding(key, value)
            
        # Try to store one more
        with pytest.raises(RuntimeError):
            qmem.store_embedding(np.random.randn(4), np.random.randn(4))
            
    def test_memory_reset(self):
        """Test memory reset functionality."""
        qmem = QuantumMemory(capacity=8, embedding_dim=4)
        
        # Store some data
        qmem.store_embedding(np.random.randn(4), np.random.randn(4))
        assert qmem.stored_count == 1
        
        # Reset
        qmem.reset()
        assert qmem.stored_count == 0
        assert np.allclose(qmem.qram.memory, 0.0)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_quantum_to_classical_conversion(self):
        """Test quantum state to classical conversion."""
        # Create a simple quantum state
        state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0])
        state = Statevector(state_vector)
        
        # Test different conversion methods
        classical_exp = quantum_state_to_classical(state, method="expectation")
        classical_amp = quantum_state_to_classical(state, method="amplitude")
        
        assert isinstance(classical_exp, np.ndarray)
        assert isinstance(classical_amp, np.ndarray)
        assert len(classical_exp) == 2  # log2(4) qubits
        assert len(classical_amp) == 4  # state dimension
        
    def test_classical_to_quantum_conversion(self):
        """Test classical to quantum state conversion."""
        classical_data = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Test different encoding methods
        state_amp = classical_to_quantum_state(classical_data, encoding="amplitude")
        state_angle = classical_to_quantum_state(classical_data[:2], encoding="angle")
        
        assert isinstance(state_amp, Statevector)
        assert isinstance(state_angle, Statevector)
        
        # Check normalization
        assert np.isclose(np.sum(np.abs(state_amp.data)**2), 1.0)
        assert np.isclose(np.sum(np.abs(state_angle.data)**2), 1.0)
        
    def test_invalid_conversion_method(self):
        """Test invalid conversion methods."""
        state = Statevector([1.0, 0.0])
        
        with pytest.raises(ValueError):
            quantum_state_to_classical(state, method="invalid")
            
        with pytest.raises(ValueError):
            classical_to_quantum_state([1.0, 0.0], encoding="invalid")


class TestIntegration:
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
            assert isinstance(retrieved, np.ndarray)
            assert len(retrieved) == 4
            
    def test_circuit_generation_and_execution(self):
        """Test quantum circuit generation and basic properties."""
        qmem = QuantumMemory(capacity=4, embedding_dim=4)
        
        # Store some data
        qmem.store_embedding(np.array([1.0, 0.0, 0.0, 0.0]), 
                           np.array([0.0, 1.0, 0.0, 0.0]))
        
        # Get circuit
        circuit = qmem.get_circuit()
        
        assert circuit is not None
        assert circuit.num_qubits > 0
        assert circuit.depth() >= 0


@pytest.fixture
def sample_qram():
    """Fixture providing a sample QRAM for testing."""
    qram = QRAM(memory_size=8, address_qubits=3)
    
    # Pre-populate with some data
    for i in range(4):
        data = np.zeros(8)
        data[i] = 1.0
        qram.write(i, data)
        
    return qram


@pytest.fixture
def sample_quantum_memory():
    """Fixture providing a sample QuantumMemory for testing."""
    qmem = QuantumMemory(capacity=8, embedding_dim=4)
    
    # Pre-populate with some embeddings
    for i in range(3):
        key = np.zeros(4)
        key[i] = 1.0
        value = np.zeros(4)
        value[(i+1) % 4] = 1.0
        qmem.store_embedding(key, value)
        
    return qmem


class TestFixtures:
    """Test cases using fixtures."""
    
    def test_sample_qram_fixture(self, sample_qram):
        """Test the sample QRAM fixture."""
        assert sample_qram.memory_size == 8
        assert sample_qram.address_qubits == 3
        
        # Check that data was stored
        for i in range(4):
            assert not np.allclose(sample_qram.memory[i], 0.0)
            
    def test_sample_quantum_memory_fixture(self, sample_quantum_memory):
        """Test the sample QuantumMemory fixture."""
        assert sample_quantum_memory.capacity == 8
        assert sample_quantum_memory.embedding_dim == 4
        assert sample_quantum_memory.stored_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
