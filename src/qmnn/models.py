"""
Quantum Memory-Augmented Neural Network models.

This module implements modular QMANN models with optimized quantum-classical integration,
vectorized operations, and proper PyTorch nn.Module compatibility.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .core import QuantumMemory


class QuantumNeuralNetwork(nn.Module):
    """
    Optimized quantum neural network with vectorized operations.

    Implements parameterized quantum circuits with efficient batch processing
    and realistic quantum hardware constraints.
    """

    def __init__(self, input_dim: int, output_dim: int, n_qubits: int = 4,
                 circuit_depth: int = 2, use_entanglement: bool = True):
        """
        Initialize quantum neural network with hardware-aware design.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            n_qubits: Number of qubits in quantum circuit (max 12 for realism)
            circuit_depth: Depth of quantum circuit layers
            use_entanglement: Whether to include entangling gates
        """
        super().__init__()

        # Validate hardware constraints
        if n_qubits > 12:
            warnings.warn(f"n_qubits={n_qubits} > 12 may exceed current hardware limits")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = min(n_qubits, 12)  # Hardware constraint
        self.circuit_depth = circuit_depth
        self.use_entanglement = use_entanglement

        # Classical preprocessing with proper scaling
        self.input_layer = nn.Linear(input_dim, self.n_qubits)
        self.input_norm = nn.LayerNorm(self.n_qubits)

        # Quantum parameters (trainable) - organized by layer and gate type
        self.rotation_params = nn.Parameter(
            torch.randn(circuit_depth, self.n_qubits, 3) * 0.1
        )  # RX, RY, RZ rotations

        if use_entanglement:
            self.entanglement_params = nn.Parameter(
                torch.randn(circuit_depth, self.n_qubits - 1) * 0.1
            )  # CNOT gate parameters

        # Classical postprocessing with residual connection
        self.output_layer = nn.Sequential(
            nn.Linear(self.n_qubits, self.n_qubits),
            nn.ReLU(),
            nn.Linear(self.n_qubits, output_dim)
        )

        # Measurement basis parameters
        self.measurement_basis = nn.Parameter(torch.randn(self.n_qubits) * 0.1)
        
    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized quantum circuit forward pass.

        Args:
            x: Input tensor (batch_size, n_qubits)

        Returns:
            Quantum circuit output (batch_size, n_qubits)
        """
        batch_size = x.shape[0]

        # Initialize quantum state amplitudes (simplified representation)
        state = torch.zeros(batch_size, self.n_qubits, device=x.device)

        # Input encoding: map classical data to quantum state
        encoded_angles = x * np.pi  # Scale to [0, π] for rotation angles

        # Apply quantum circuit layers
        for layer in range(self.circuit_depth):
            # Single-qubit rotations (vectorized)
            rx_angles = encoded_angles + self.rotation_params[layer, :, 0]
            ry_angles = self.rotation_params[layer, :, 1]
            rz_angles = self.rotation_params[layer, :, 2]

            # Apply rotations (simplified quantum state evolution)
            state = state * torch.cos(rx_angles / 2) + torch.sin(ry_angles / 2)
            state = state * torch.exp(1j * rz_angles / 2).real  # Keep real for simplicity

            # Entangling gates (if enabled)
            if self.use_entanglement and layer < self.circuit_depth - 1:
                state = self._apply_entanglement(state, layer)

        # Measurement simulation
        measured_state = self._simulate_measurement(state)

        return measured_state

    def _apply_entanglement(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Apply entangling gates between adjacent qubits.

        Args:
            state: Current quantum state
            layer: Current circuit layer

        Returns:
            State after entanglement
        """
        entangled_state = state.clone()

        # Apply CNOT-like operations between adjacent qubits
        for i in range(self.n_qubits - 1):
            control_strength = torch.sigmoid(self.entanglement_params[layer, i])

            # Simplified entanglement: mix adjacent qubit states
            entangled_state[:, i] = (
                state[:, i] * (1 - control_strength) +
                state[:, i + 1] * control_strength
            )
            entangled_state[:, i + 1] = (
                state[:, i + 1] * (1 - control_strength) +
                state[:, i] * control_strength
            )

        return entangled_state

    def _simulate_measurement(self, state: torch.Tensor) -> torch.Tensor:
        """
        Simulate quantum measurement in computational basis.

        Args:
            state: Quantum state to measure

        Returns:
            Measurement outcomes
        """
        # Apply measurement basis rotation
        basis_rotation = torch.tanh(self.measurement_basis)
        measured = state * basis_rotation

        # Add quantum noise (decoherence simulation)
        noise_level = 0.01
        noise = torch.randn_like(measured) * noise_level
        measured = measured + noise

        # Normalize to simulate probability amplitudes
        measured = torch.tanh(measured)  # Keep in [-1, 1] range

        return measured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum neural network.

        Args:
            x: Input tensor

        Returns:
            Network output
        """
        # Classical preprocessing with normalization
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = torch.tanh(x)  # Normalize for quantum encoding

        # Quantum processing
        x = self.quantum_forward(x)

        # Classical postprocessing
        x = self.output_layer(x)

        return x

    def get_quantum_info(self) -> Dict[str, Any]:
        """
        Get information about the quantum circuit.

        Returns:
            Dictionary with quantum circuit information
        """
        return {
            'n_qubits': self.n_qubits,
            'circuit_depth': self.circuit_depth,
            'use_entanglement': self.use_entanglement,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'quantum_parameters': self.rotation_params.numel() + (
                self.entanglement_params.numel() if self.use_entanglement else 0
            ),
            'classical_parameters': (
                self.input_layer.weight.numel() + self.input_layer.bias.numel() +
                sum(p.numel() for p in self.output_layer.parameters())
            )
        }


class QMANN(nn.Module):
    """
    Optimized Quantum Memory-Augmented Neural Network.

    Combines classical neural networks with quantum memory using modular design,
    vectorized operations, and hardware-aware optimizations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        memory_capacity: int = 256,
        memory_embedding_dim: int = 64,
        n_quantum_layers: int = 2,
        max_qubits: int = 16,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize QMANN with modular architecture.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            memory_capacity: Quantum memory capacity
            memory_embedding_dim: Memory embedding dimension
            n_quantum_layers: Number of quantum processing layers
            max_qubits: Maximum qubits for quantum components
            use_attention: Whether to use attention mechanisms
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # Validate and store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_capacity = memory_capacity
        self.memory_embedding_dim = memory_embedding_dim
        self.max_qubits = max_qubits
        self.use_attention = use_attention
        self.dropout = dropout

        # Calculate optimal quantum memory configuration
        effective_capacity = min(memory_capacity, 2**(max_qubits // 2))
        if effective_capacity < memory_capacity:
            warnings.warn(f"Memory capacity reduced to {effective_capacity} due to qubit constraints")

        # Classical encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, memory_embedding_dim),
            nn.LayerNorm(memory_embedding_dim),
        )

        # LSTM controller with improved architecture
        self.controller = nn.LSTM(
            input_size=memory_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if n_quantum_layers > 1 else 0,
            bidirectional=False  # Keep unidirectional for memory consistency
        )

        # Quantum memory with hardware constraints
        self.quantum_memory = QuantumMemory(
            capacity=effective_capacity,
            embedding_dim=memory_embedding_dim,
            max_qubits=max_qubits,
            use_amplitude_encoding=True
        )

        # Quantum processing layers with optimized qubit allocation
        quantum_qubits = min(8, memory_embedding_dim, max_qubits // 2)
        self.quantum_layers = nn.ModuleList([
            QuantumNeuralNetwork(
                input_dim=memory_embedding_dim,
                output_dim=memory_embedding_dim,
                n_qubits=quantum_qubits,
                circuit_depth=2,
                use_entanglement=True
            )
            for _ in range(n_quantum_layers)
        ])

        # Memory attention mechanism (optional)
        if use_attention:
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=memory_embedding_dim,
                num_heads=min(8, memory_embedding_dim // 8),
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.memory_attention = None

        # Output decoder with skip connections
        decoder_input_dim = hidden_dim + memory_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Memory management parameters
        self.memory_write_threshold = nn.Parameter(torch.tensor(0.5))
        self.memory_read_temperature = nn.Parameter(torch.tensor(1.0))

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to memory embedding space."""
        return self.encoder(x)

    def quantum_memory_read(self, query: torch.Tensor,
                          similarity_threshold: float = 0.1) -> torch.Tensor:
        """
        Vectorized quantum memory read operation.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            similarity_threshold: Minimum similarity for memory retrieval

        Returns:
            Retrieved memory content [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = query.shape
        device = query.device

        # Flatten for batch processing
        flat_query = query.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]

        # Convert to numpy for quantum memory interface
        query_np = flat_query.detach().cpu().numpy()

        # Batch retrieve from quantum memory
        try:
            retrieved_np = self.quantum_memory.batch_retrieve(
                query_np, similarity_threshold
            )
            retrieved = torch.from_numpy(retrieved_np).float().to(device)
        except Exception as e:
            # Fallback to zero tensor if quantum memory fails
            warnings.warn(f"Quantum memory read failed: {e}, using zero fallback")
            retrieved = torch.zeros_like(flat_query)

        # Reshape back to original dimensions
        retrieved = retrieved.view(batch_size, seq_len, embed_dim)

        # Apply temperature scaling for smoother retrieval
        temperature = torch.sigmoid(self.memory_read_temperature)
        retrieved = retrieved * temperature

        return retrieved

    def quantum_memory_write(self, key: torch.Tensor, value: torch.Tensor,
                           write_probability: Optional[torch.Tensor] = None) -> int:
        """
        Selective quantum memory write operation.

        Args:
            key: Memory key [batch_size, seq_len, embed_dim]
            value: Memory value [batch_size, seq_len, embed_dim]
            write_probability: Optional write probability per sample

        Returns:
            Number of successful writes
        """
        batch_size, seq_len, embed_dim = key.shape
        successful_writes = 0

        # Determine which samples to write based on threshold
        if write_probability is None:
            # Use learned threshold
            write_mask = torch.rand(batch_size, seq_len, device=key.device) > torch.sigmoid(self.memory_write_threshold)
        else:
            write_mask = torch.rand(batch_size, seq_len, device=key.device) < write_probability

        # Write selected samples to memory
        for b in range(batch_size):
            for s in range(seq_len):
                if write_mask[b, s]:
                    try:
                        key_np = key[b, s].detach().cpu().numpy()
                        value_np = value[b, s].detach().cpu().numpy()
                        self.quantum_memory.store_embedding(key_np, value_np)
                        successful_writes += 1
                    except RuntimeError:
                        # Memory full - skip this write
                        continue

        return successful_writes

    def forward(
        self,
        x: torch.Tensor,
        store_memory: bool = True,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Optimized forward pass through QMANN.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            store_memory: Whether to store intermediate states in memory
            return_attention: Whether to return attention weights

        Returns:
            output: Network output [batch_size, seq_len, output_dim]
            memory_content: Retrieved memory content (if return_attention=True)
            attention_weights: Attention weights (if return_attention=True and use_attention=True)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Encode input to memory embedding space
        encoded = self.encode_input(x)  # [batch_size, seq_len, memory_embedding_dim]

        # 2. Controller processing with LSTM
        controller_out, (h_n, c_n) = self.controller(encoded)

        # 3. Quantum memory read operation
        memory_content = self.quantum_memory_read(
            query=encoded,
            similarity_threshold=0.1
        )

        # 4. Apply quantum processing layers sequentially
        quantum_processed = memory_content
        for i, quantum_layer in enumerate(self.quantum_layers):
            # Reshape for quantum layer processing
            flat_input = quantum_processed.view(-1, quantum_processed.shape[-1])
            quantum_out = quantum_layer(flat_input)
            quantum_processed = quantum_out.view(quantum_processed.shape)

            # Add residual connection for deeper networks
            if i > 0:
                quantum_processed = quantum_processed + memory_content

        # 5. Memory attention mechanism (if enabled)
        attention_weights = None
        if self.use_attention and self.memory_attention is not None:
            attended_memory, attention_weights = self.memory_attention(
                query=controller_out,
                key=quantum_processed,
                value=quantum_processed,
            )
        else:
            # Simple weighted combination without attention
            attended_memory = quantum_processed

        # 6. Combine controller output with processed memory
        combined = torch.cat([controller_out, attended_memory], dim=-1)

        # 7. Decode to final output
        output = self.decoder(combined)

        # 8. Selective memory storage during training
        if store_memory and self.training:
            # Only store a subset of samples to avoid memory overflow
            write_prob = torch.sigmoid(self.memory_write_threshold)
            self.quantum_memory_write(encoded, controller_out, write_prob)

        # Return based on requested outputs
        if return_attention:
            if attention_weights is not None:
                return output, attended_memory, attention_weights
            else:
                return output, attended_memory
        else:
            return output

    def reset_memory(self) -> None:
        """Reset quantum memory contents."""
        self.quantum_memory.reset()

    def get_memory_circuit(self):
        """Get the quantum memory circuit for visualization."""
        return self.quantum_memory.get_circuit()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model architecture and performance info
        """
        memory_info = self.quantum_memory.get_memory_info()

        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        classical_params = (
            sum(p.numel() for p in self.encoder.parameters()) +
            sum(p.numel() for p in self.controller.parameters()) +
            sum(p.numel() for p in self.decoder.parameters())
        )
        quantum_params = sum(p.numel() for p in self.quantum_layers.parameters())

        if self.memory_attention is not None:
            attention_params = sum(p.numel() for p in self.memory_attention.parameters())
            classical_params += attention_params
        else:
            attention_params = 0

        return {
            'model_type': 'QMANN',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'memory_embedding_dim': self.memory_embedding_dim,
            'total_parameters': total_params,
            'classical_parameters': classical_params,
            'quantum_parameters': quantum_params,
            'attention_parameters': attention_params,
            'quantum_layers': len(self.quantum_layers),
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout,
            'memory_info': memory_info,
            'hardware_requirements': {
                'max_qubits': self.max_qubits,
                'quantum_circuit_depth': sum(
                    layer.circuit_depth for layer in self.quantum_layers
                ),
                'memory_qubits': memory_info.get('total_qubits', 0)
            }
        }

    def memory_usage(self) -> float:
        """Get current quantum memory usage ratio."""
        return self.quantum_memory.memory_usage()

    def optimize_memory(self) -> None:
        """
        Optimize quantum memory by defragmentation.

        This should be called periodically during training to maintain
        memory efficiency.
        """
        self.quantum_memory.defragment()

    def validate_hardware_constraints(self) -> Dict[str, bool]:
        """
        Validate that the model meets quantum hardware constraints.

        Returns:
            Dictionary of constraint validation results
        """
        memory_constraints = self.quantum_memory.validate_quantum_constraints()

        # Additional model-level constraints
        total_qubits = sum(layer.n_qubits for layer in self.quantum_layers)
        total_qubits += self.quantum_memory.get_memory_info().get('total_qubits', 0)

        model_constraints = {
            'total_model_qubits_feasible': total_qubits <= 50,  # Conservative estimate
            'quantum_layers_reasonable': len(self.quantum_layers) <= 5,
            'memory_capacity_practical': self.memory_capacity <= 512,
            'embedding_dim_suitable': self.memory_embedding_dim <= 128
        }

        return {**memory_constraints, **model_constraints}
        
    def memory_usage(self) -> float:
        """Get current memory usage ratio."""
        return self.quantum_memory.stored_count / self.quantum_memory.capacity
