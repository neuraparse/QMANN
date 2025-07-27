"""
Unit tests for QMANN models.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from qmann.models import QuantumNeuralNetwork, QMANN


class TestQuantumNeuralNetwork:
    """Test cases for QuantumNeuralNetwork."""
    
    def test_qnn_initialization(self):
        """Test QNN initialization with valid parameters."""
        input_dim = 10
        output_dim = 5
        n_qubits = 4
        
        qnn = QuantumNeuralNetwork(input_dim, output_dim, n_qubits)
        
        assert qnn.input_dim == input_dim
        assert qnn.output_dim == output_dim
        assert qnn.n_qubits == n_qubits
        assert isinstance(qnn.input_layer, nn.Linear)
        assert isinstance(qnn.output_layer, nn.Linear)
        assert qnn.quantum_params.shape == (n_qubits, 3)
        
    def test_qnn_forward_pass(self):
        """Test QNN forward pass."""
        qnn = QuantumNeuralNetwork(input_dim=8, output_dim=3, n_qubits=4)
        
        # Create sample input
        batch_size = 2
        x = torch.randn(batch_size, 8)
        
        # Forward pass
        output = qnn(x)
        
        assert output.shape == (batch_size, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_qnn_quantum_forward(self):
        """Test quantum forward pass simulation."""
        qnn = QuantumNeuralNetwork(input_dim=4, output_dim=2, n_qubits=4)
        
        # Create normalized input
        x = torch.randn(3, 4)
        x = torch.tanh(x)  # Normalize for quantum encoding
        
        quantum_out = qnn.quantum_forward(x)
        
        assert quantum_out.shape == (3, 4)
        assert not torch.isnan(quantum_out).any()
        
    def test_qnn_gradient_flow(self):
        """Test gradient flow through QNN."""
        qnn = QuantumNeuralNetwork(input_dim=4, output_dim=2, n_qubits=4)
        
        x = torch.randn(2, 4, requires_grad=True)
        y_true = torch.randint(0, 2, (2,))
        
        # Forward pass
        y_pred = qnn(x)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert qnn.quantum_params.grad is not None
        assert not torch.isnan(qnn.quantum_params.grad).any()


class TestQMANN:
    """Test cases for QMANN."""
    
    def test_qmann_initialization(self):
        """Test QMANN initialization."""
        input_dim = 10
        hidden_dim = 64
        output_dim = 5
        memory_capacity = 128
        memory_embedding_dim = 32
        
        qmann = QMANN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            memory_capacity=memory_capacity,
            memory_embedding_dim=memory_embedding_dim,
        )
        
        assert qmann.input_dim == input_dim
        assert qmann.hidden_dim == hidden_dim
        assert qmann.output_dim == output_dim
        assert qmann.memory_capacity == memory_capacity
        assert qmann.memory_embedding_dim == memory_embedding_dim
        
        # Check components
        assert isinstance(qmann.encoder, nn.Sequential)
        assert isinstance(qmann.controller, nn.LSTM)
        assert qmann.quantum_memory is not None
        assert len(qmann.quantum_layers) > 0
        assert isinstance(qmann.decoder, nn.Sequential)
        
    def test_qmann_forward_pass(self):
        """Test QMANN forward pass."""
        qmann = QMANN(
            input_dim=8,
            hidden_dim=32,
            output_dim=3,
            memory_capacity=64,
            memory_embedding_dim=16,
        )
        
        # Create sample input [batch_size, seq_len, input_dim]
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, 8)
        
        # Forward pass
        output, memory_content = qmann(x)
        
        assert output.shape == (batch_size, seq_len, 3)
        assert memory_content.shape == (batch_size, seq_len, 16)
        assert not torch.isnan(output).any()
        assert not torch.isnan(memory_content).any()
        
    def test_qmann_encode_input(self):
        """Test input encoding."""
        qmann = QMANN(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            memory_capacity=64,
            memory_embedding_dim=16,
        )
        
        x = torch.randn(3, 4, 10)
        encoded = qmann.encode_input(x)
        
        assert encoded.shape == (3, 4, 16)
        assert not torch.isnan(encoded).any()
        
    def test_qmann_memory_operations(self):
        """Test quantum memory read/write operations."""
        qmann = QMANN(
            input_dim=8,
            hidden_dim=32,
            output_dim=3,
            memory_capacity=32,
            memory_embedding_dim=16,
        )
        
        # Test memory write
        key = torch.randn(2, 3, 16)
        value = torch.randn(2, 3, 32)
        
        # This should not raise an error
        qmann.quantum_memory_write(key, value)
        
        # Test memory read
        query = torch.randn(2, 3, 16)
        retrieved = qmann.quantum_memory_read(query)
        
        assert retrieved.shape == (2, 3, 16)
        assert not torch.isnan(retrieved).any()
        
    def test_qmann_memory_usage(self):
        """Test memory usage tracking."""
        qmann = QMANN(
            input_dim=8,
            hidden_dim=32,
            output_dim=3,
            memory_capacity=16,
            memory_embedding_dim=8,
        )
        
        initial_usage = qmann.memory_usage()
        assert initial_usage == 0.0
        
        # Store some data
        key = torch.randn(1, 1, 8)
        value = torch.randn(1, 1, 32)
        qmann.quantum_memory_write(key, value)
        
        updated_usage = qmann.memory_usage()
        assert updated_usage > initial_usage
        assert 0.0 <= updated_usage <= 1.0
        
    def test_qmann_reset_memory(self):
        """Test memory reset functionality."""
        qmann = QMANN(
            input_dim=8,
            hidden_dim=32,
            output_dim=3,
            memory_capacity=16,
            memory_embedding_dim=8,
        )
        
        # Store some data
        key = torch.randn(1, 1, 8)
        value = torch.randn(1, 1, 32)
        qmann.quantum_memory_write(key, value)
        
        assert qmann.memory_usage() > 0.0
        
        # Reset memory
        qmann.reset_memory()
        
        assert qmann.memory_usage() == 0.0
        
    def test_qmann_gradient_flow(self):
        """Test gradient flow through QMANN."""
        qmann = QMANN(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            memory_capacity=8,
            memory_embedding_dim=8,
        )
        
        x = torch.randn(1, 3, 4, requires_grad=True)
        y_true = torch.randint(0, 2, (1, 3))
        
        # Forward pass
        y_pred, _ = qmann(x)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(y_pred.view(-1, 2), y_true.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        
        # Check that model parameters have gradients
        for param in qmann.parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_qmann_training_mode(self):
        """Test QMANN behavior in training vs evaluation mode."""
        qmann = QMANN(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            memory_capacity=8,
            memory_embedding_dim=8,
        )
        
        x = torch.randn(1, 2, 4)
        
        # Training mode
        qmann.train()
        output_train, _ = qmann(x, store_memory=True)
        
        # Evaluation mode
        qmann.eval()
        with torch.no_grad():
            output_eval, _ = qmann(x, store_memory=False)
            
        assert output_train.shape == output_eval.shape
        # Outputs may differ due to memory storage in training mode
        
    def test_qmann_get_memory_circuit(self):
        """Test getting quantum memory circuit."""
        qmann = QMANN(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            memory_capacity=8,
            memory_embedding_dim=8,
        )
        
        circuit = qmann.get_memory_circuit()
        assert circuit is not None
        assert circuit.num_qubits > 0


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_qnn_qmann_integration(self):
        """Test integration between QNN and QMANN components."""
        qmann = QMANN(
            input_dim=6,
            hidden_dim=24,
            output_dim=3,
            memory_capacity=16,
            memory_embedding_dim=12,
            n_quantum_layers=2,
        )
        
        # Test that quantum layers are properly integrated
        assert len(qmann.quantum_layers) == 2
        
        for layer in qmann.quantum_layers:
            assert isinstance(layer, QuantumNeuralNetwork)
            
        # Test forward pass through integrated system
        x = torch.randn(2, 4, 6)
        output, memory = qmann(x)
        
        assert output.shape == (2, 4, 3)
        assert memory.shape == (2, 4, 12)
        
    def test_memory_attention_integration(self):
        """Test memory attention mechanism."""
        qmann = QMANN(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            memory_capacity=8,
            memory_embedding_dim=8,
        )
        
        # Test that attention mechanism works
        x = torch.randn(1, 3, 4)
        
        # Forward pass should use attention
        output, attended_memory = qmann(x)
        
        assert attended_memory.shape == (1, 3, 8)
        assert not torch.isnan(attended_memory).any()


@pytest.fixture
def sample_qnn():
    """Fixture providing a sample QNN for testing."""
    return QuantumNeuralNetwork(input_dim=4, output_dim=2, n_qubits=4)


@pytest.fixture
def sample_qmann():
    """Fixture providing a sample QMANN for testing."""
    return QMANN(
        input_dim=6,
        hidden_dim=24,
        output_dim=3,
        memory_capacity=16,
        memory_embedding_dim=12,
    )


class TestFixtures:
    """Test cases using fixtures."""
    
    def test_sample_qnn_fixture(self, sample_qnn):
        """Test the sample QNN fixture."""
        assert sample_qnn.input_dim == 4
        assert sample_qnn.output_dim == 2
        assert sample_qnn.n_qubits == 4
        
        # Test forward pass
        x = torch.randn(2, 4)
        output = sample_qnn(x)
        assert output.shape == (2, 2)
        
    def test_sample_qmann_fixture(self, sample_qmann):
        """Test the sample QMANN fixture."""
        assert sample_qmann.input_dim == 6
        assert sample_qmann.output_dim == 3
        assert sample_qmann.memory_capacity == 16
        
        # Test forward pass
        x = torch.randn(1, 3, 6)
        output, memory = sample_qmann(x)
        assert output.shape == (1, 3, 3)
        assert memory.shape == (1, 3, 12)


if __name__ == "__main__":
    pytest.main([__file__])
