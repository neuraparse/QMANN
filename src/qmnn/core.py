"""
Core quantum memory components for QMANN.

This module implements realistic quantum random access memory (QRAM)
with amplitude encoding, vectorized operations, and proper scaling limitations.
"""

from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import warnings
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import pennylane as qml


class QRAM:
    """
    Realistic Quantum Random Access Memory implementation.

    Provides quantum superposition-based memory access with amplitude encoding
    and proper scaling limitations for near-term quantum devices.

    Note: Current implementation uses classical simulation of quantum operations.
    Real quantum hardware limitations apply for practical use.
    """

    def __init__(self, memory_size: int, address_qubits: int,
                 max_data_qubits: int = 8, use_amplitude_encoding: bool = True):
        """
        Initialize QRAM with realistic constraints.

        Args:
            memory_size: Number of memory cells (limited by address_qubits)
            address_qubits: Number of qubits for addressing (practical limit ~10)
            max_data_qubits: Maximum data qubits per cell (hardware constraint)
            use_amplitude_encoding: Whether to use amplitude encoding for data
        """
        # Validate realistic constraints
        if address_qubits > 10:
            warnings.warn(f"Address qubits {address_qubits} > 10 may not be practical on current hardware")

        if max_data_qubits > 12:
            warnings.warn(f"Data qubits {max_data_qubits} > 12 may exceed current hardware limits")

        self.memory_size = min(memory_size, 2**address_qubits)
        self.address_qubits = address_qubits
        self.max_data_qubits = max_data_qubits
        self.use_amplitude_encoding = use_amplitude_encoding

        # Calculate effective data dimension
        if use_amplitude_encoding:
            # Amplitude encoding: can store 2^n values in n qubits
            self.data_dimension = 2**max_data_qubits
        else:
            # Basis encoding: can store n values in n qubits
            self.data_dimension = max_data_qubits

        # Initialize memory with amplitude-encoded states
        self.memory = np.zeros((self.memory_size, self.data_dimension), dtype=complex)
        self._circuit = None
        self._memory_usage = 0.0

    def write(self, address: int, data: np.ndarray) -> None:
        """
        Write classical data to quantum memory using amplitude encoding.

        Args:
            address: Memory address
            data: Classical data vector to store
        """
        if address >= self.memory_size:
            raise ValueError(f"Address {address} out of bounds")

        if self.use_amplitude_encoding:
            # Amplitude encoding: encode data as quantum state amplitudes
            encoded_data = self._amplitude_encode(data)
        else:
            # Basis encoding: direct storage with normalization
            encoded_data = self._basis_encode(data)

        self.memory[address] = encoded_data
        self._memory_usage = np.sum(np.abs(self.memory) > 1e-10) / self.memory.size

    def _amplitude_encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode classical data as quantum state amplitudes.

        Args:
            data: Classical data vector

        Returns:
            Amplitude-encoded quantum state
        """
        # Pad or truncate data to fit quantum state dimension
        if len(data) > self.data_dimension:
            # Truncate with warning
            warnings.warn(f"Data dimension {len(data)} > {self.data_dimension}, truncating")
            data = data[:self.data_dimension]
        elif len(data) < self.data_dimension:
            # Pad with zeros
            padded_data = np.zeros(self.data_dimension)
            padded_data[:len(data)] = data
            data = padded_data

        # Normalize for valid quantum state
        norm = np.linalg.norm(data)
        if norm > 1e-10:
            return data / norm
        else:
            # Return uniform superposition for zero data
            return np.ones(self.data_dimension) / np.sqrt(self.data_dimension)

    def _basis_encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode classical data using basis encoding.

        Args:
            data: Classical data vector

        Returns:
            Basis-encoded state
        """
        # Simple normalization for basis encoding
        encoded = np.zeros(self.data_dimension, dtype=complex)
        data_len = min(len(data), self.data_dimension)
        encoded[:data_len] = data[:data_len]

        # Normalize
        norm = np.linalg.norm(encoded)
        if norm > 1e-10:
            encoded /= norm

        return encoded

    def read(self, address_state: np.ndarray) -> Statevector:
        """
        Vectorized quantum read operation with superposition addressing.

        Args:
            address_state: Quantum superposition of addresses

        Returns:
            Quantum state vector containing superposed memory contents
        """
        if len(address_state) != 2**self.address_qubits:
            raise ValueError(f"Address state dimension {len(address_state)} != {2**self.address_qubits}")

        # Vectorized computation for efficiency
        valid_addresses = min(self.memory_size, len(address_state))

        # Extract valid amplitudes and memory states
        valid_amplitudes = address_state[:valid_addresses]
        valid_memory = self.memory[:valid_addresses]

        # Vectorized superposition computation
        result_state = np.sum(valid_amplitudes[:, np.newaxis] * valid_memory, axis=0)

        # Ensure proper normalization
        norm = np.linalg.norm(result_state)
        if norm > 1e-10:
            result_state /= norm

        return Statevector(result_state)

    def batch_read(self, address_states: np.ndarray) -> List[Statevector]:
        """
        Batch read operation for multiple address superpositions.

        Args:
            address_states: Array of address superpositions [batch_size, 2^address_qubits]

        Returns:
            List of quantum state vectors
        """
        results = []
        for address_state in address_states:
            results.append(self.read(address_state))
        return results

    def create_circuit(self) -> QuantumCircuit:
        """
        Create realistic quantum circuit for QRAM operations.

        Note: This creates a simplified circuit suitable for near-term devices.
        Full QRAM implementation requires fault-tolerant quantum computers.

        Returns:
            Quantum circuit implementing simplified QRAM
        """
        total_qubits = self.address_qubits + self.max_data_qubits
        qc = QuantumCircuit(total_qubits)

        # Address register
        addr_reg = QuantumRegister(self.address_qubits, 'addr')
        # Data register
        data_reg = QuantumRegister(self.max_data_qubits, 'data')

        qc.add_register(addr_reg)
        qc.add_register(data_reg)

        # Implement simplified QRAM using controlled rotations
        # Note: This is a classical simulation of quantum operations
        for addr in range(min(self.memory_size, 2**self.address_qubits)):
            if not np.any(np.abs(self.memory[addr]) > 1e-10):
                continue  # Skip empty memory cells

            # Create address selector pattern
            addr_binary = format(addr, f'0{self.address_qubits}b')

            # Apply X gates for 0 bits in address (creates selector)
            for i, bit in enumerate(addr_binary):
                if bit == '0':
                    qc.x(addr_reg[i])

            # Encode memory content using controlled operations
            if self.use_amplitude_encoding:
                self._add_amplitude_encoding_gates(qc, addr_reg, data_reg, addr)
            else:
                self._add_basis_encoding_gates(qc, addr_reg, data_reg, addr)

            # Restore address qubits
            for i, bit in enumerate(addr_binary):
                if bit == '0':
                    qc.x(addr_reg[i])

        self._circuit = qc
        return qc

    def _add_amplitude_encoding_gates(self, qc: QuantumCircuit,
                                    addr_reg: QuantumRegister,
                                    data_reg: QuantumRegister,
                                    addr: int) -> None:
        """Add amplitude encoding gates for specific address."""
        memory_data = self.memory[addr]

        # Use controlled rotations to encode amplitudes
        for i, amplitude in enumerate(memory_data[:2**self.max_data_qubits]):
            if abs(amplitude) > 1e-10:
                # Convert amplitude to rotation angle
                angle = 2 * np.arcsin(min(abs(amplitude), 1.0))

                # Apply multi-controlled rotation
                if i < self.max_data_qubits:
                    # Single qubit rotation
                    qc.mcry(angle, addr_reg, data_reg[i])
                else:
                    # For higher dimensions, use combinations of qubits
                    qubit_pattern = format(i, f'0{self.max_data_qubits}b')
                    target_qubits = [j for j, bit in enumerate(qubit_pattern) if bit == '1']
                    if target_qubits:
                        qc.mcry(angle, addr_reg, data_reg[target_qubits[0]])

    def _add_basis_encoding_gates(self, qc: QuantumCircuit,
                                addr_reg: QuantumRegister,
                                data_reg: QuantumRegister,
                                addr: int) -> None:
        """Add basis encoding gates for specific address."""
        memory_data = self.memory[addr]

        # Use controlled X gates for basis encoding
        for i, value in enumerate(memory_data[:self.max_data_qubits]):
            if abs(value) > 0.5:  # Threshold for basis state
                qc.mcx(addr_reg, data_reg[i])

    def capacity_bound(self) -> Dict[str, float]:
        """
        Calculate realistic and theoretical capacity bounds.

        Returns:
            Dictionary with capacity information
        """
        theoretical_capacity = 2**self.address_qubits * self.data_dimension

        # Practical capacity considering hardware limitations
        practical_capacity = min(
            theoretical_capacity,
            self.memory_size * self.data_dimension
        )

        # Effective capacity considering encoding efficiency
        if self.use_amplitude_encoding:
            encoding_efficiency = 1.0  # Full utilization of quantum state space
        else:
            encoding_efficiency = 0.5  # Basis encoding is less efficient

        effective_capacity = practical_capacity * encoding_efficiency

        return {
            'theoretical_bits': theoretical_capacity,
            'practical_bits': practical_capacity,
            'effective_bits': effective_capacity,
            'memory_usage_ratio': self._memory_usage,
            'encoding_type': 'amplitude' if self.use_amplitude_encoding else 'basis'
        }

    def get_hardware_requirements(self) -> Dict[str, int]:
        """
        Get hardware requirements for this QRAM configuration.

        Returns:
            Dictionary with hardware requirements
        """
        return {
            'total_qubits': self.address_qubits + self.max_data_qubits,
            'address_qubits': self.address_qubits,
            'data_qubits': self.max_data_qubits,
            'circuit_depth': self._estimate_circuit_depth(),
            'gate_count': self._estimate_gate_count(),
            'memory_cells': self.memory_size
        }

    def _estimate_circuit_depth(self) -> int:
        """Estimate quantum circuit depth."""
        # Depth scales with memory size and encoding complexity
        base_depth = self.memory_size * 2  # Address selection + restoration
        encoding_depth = self.max_data_qubits * 2  # Encoding operations
        return base_depth + encoding_depth

    def _estimate_gate_count(self) -> int:
        """Estimate total quantum gate count."""
        # Gates per memory cell: address selection + encoding + restoration
        gates_per_cell = (
            self.address_qubits * 2 +  # X gates for address selection
            self.max_data_qubits * 2   # Encoding gates
        )
        return self.memory_size * gates_per_cell


class QuantumMemory:
    """
    High-level quantum memory interface for neural networks.

    Provides realistic quantum memory operations with proper scaling
    and hardware-aware optimizations.
    """

    def __init__(self, capacity: int, embedding_dim: int,
                 max_qubits: int = 16, use_amplitude_encoding: bool = True):
        """
        Initialize quantum memory with realistic constraints.

        Args:
            capacity: Memory capacity (number of entries)
            embedding_dim: Dimension of stored embeddings
            max_qubits: Maximum total qubits available (hardware constraint)
            use_amplitude_encoding: Whether to use amplitude encoding
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.max_qubits = max_qubits

        # Calculate optimal qubit allocation
        self.address_qubits = min(int(np.ceil(np.log2(capacity))), max_qubits // 2)
        self.data_qubits = min(max_qubits - self.address_qubits, 12)  # Hardware limit

        # Adjust capacity to fit address space
        self.effective_capacity = min(capacity, 2**self.address_qubits)

        # Initialize QRAM with constraints
        self.qram = QRAM(
            memory_size=self.effective_capacity,
            address_qubits=self.address_qubits,
            max_data_qubits=self.data_qubits,
            use_amplitude_encoding=use_amplitude_encoding
        )

        self.stored_count = 0
        self._key_to_address = {}  # Key-address mapping for associative memory

    def store_embedding(self, key: np.ndarray, value: np.ndarray) -> int:
        """
        Store key-value embedding pair with collision handling.

        Args:
            key: Key vector for addressing
            value: Value vector to store

        Returns:
            Storage address
        """
        if self.stored_count >= self.effective_capacity:
            # Implement LRU replacement or raise error
            warnings.warn(f"Memory capacity {self.effective_capacity} exceeded, overwriting oldest entry")
            address = self._find_lru_address()
        else:
            # Find available address using improved hashing
            address = self._compute_address(key)

        # Handle collisions with linear probing
        original_address = address
        while address in self._key_to_address.values() and self._key_to_address:
            address = (address + 1) % self.effective_capacity
            if address == original_address:
                raise RuntimeError("Memory full - no available addresses")

        # Store value and update mappings
        self.qram.write(address, value)
        key_hash = hash(key.tobytes())
        self._key_to_address[key_hash] = address

        if self.stored_count < self.effective_capacity:
            self.stored_count += 1

        return address

    def _compute_address(self, key: np.ndarray) -> int:
        """Compute address from key using improved hashing."""
        # Use multiple hash functions for better distribution
        hash1 = hash(key.tobytes()) % self.effective_capacity
        hash2 = hash(key.astype(np.float32).tobytes()) % self.effective_capacity

        # Combine hashes
        combined_hash = (hash1 + hash2) % self.effective_capacity
        return combined_hash

    def _find_lru_address(self) -> int:
        """Find least recently used address for replacement."""
        # Simplified LRU: return first address (in practice, would track access times)
        return 0

    def retrieve_embedding(self, query: np.ndarray, similarity_threshold: float = 0.1) -> np.ndarray:
        """
        Retrieve embedding using quantum associative memory with improved similarity.

        Args:
            query: Query vector
            similarity_threshold: Minimum similarity for address inclusion

        Returns:
            Retrieved embedding vector
        """
        # Vectorized similarity computation
        address_amplitudes = self._compute_address_amplitudes(query, similarity_threshold)

        # Quantum read with superposed addressing
        result_state = self.qram.read(address_amplitudes)

        # Extract classical embedding with proper dimension handling
        result_data = result_state.data
        if len(result_data) >= self.embedding_dim:
            return np.real(result_data[:self.embedding_dim])
        else:
            # Pad if necessary
            padded_result = np.zeros(self.embedding_dim)
            padded_result[:len(result_data)] = np.real(result_data)
            return padded_result

    def _compute_address_amplitudes(self, query: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute address amplitudes based on query similarity.

        Args:
            query: Query vector
            threshold: Similarity threshold

        Returns:
            Normalized address amplitudes
        """
        address_amplitudes = np.zeros(2**self.address_qubits)

        # Vectorized similarity computation for efficiency
        for addr in range(self.effective_capacity):
            memory_content = self.qram.memory[addr]
            if np.any(np.abs(memory_content) > 1e-10):
                # Compute cosine similarity
                query_norm = np.linalg.norm(query)
                memory_norm = np.linalg.norm(memory_content)

                if query_norm > 1e-10 and memory_norm > 1e-10:
                    # Adjust dimensions for comparison
                    min_dim = min(len(query), len(memory_content))
                    similarity = np.abs(np.dot(
                        query[:min_dim] / query_norm,
                        memory_content[:min_dim] / memory_norm
                    ))

                    if similarity > threshold:
                        address_amplitudes[addr] = similarity

        # Normalize amplitudes for valid quantum state
        norm = np.linalg.norm(address_amplitudes)
        if norm > 1e-10:
            return address_amplitudes / norm
        else:
            # Return uniform superposition if no similarities found
            uniform_amplitude = 1.0 / np.sqrt(self.effective_capacity)
            address_amplitudes[:self.effective_capacity] = uniform_amplitude
            return address_amplitudes

    def batch_retrieve(self, queries: np.ndarray, similarity_threshold: float = 0.1) -> np.ndarray:
        """
        Batch retrieve embeddings for multiple queries.

        Args:
            queries: Array of query vectors [batch_size, embedding_dim]
            similarity_threshold: Minimum similarity for address inclusion

        Returns:
            Array of retrieved embeddings [batch_size, embedding_dim]
        """
        batch_size = queries.shape[0]
        results = np.zeros((batch_size, self.embedding_dim))

        for i, query in enumerate(queries):
            results[i] = self.retrieve_embedding(query, similarity_threshold)

        return results

    def get_circuit(self) -> QuantumCircuit:
        """Get the underlying quantum circuit."""
        return self.qram.create_circuit()

    def get_memory_info(self) -> Dict[str, Union[int, float, str]]:
        """
        Get comprehensive memory information.

        Returns:
            Dictionary with memory statistics
        """
        capacity_info = self.qram.capacity_bound()
        hardware_info = self.qram.get_hardware_requirements()

        return {
            **capacity_info,
            **hardware_info,
            'stored_entries': self.stored_count,
            'capacity_utilization': self.stored_count / self.effective_capacity,
            'embedding_dimension': self.embedding_dim,
            'effective_capacity': self.effective_capacity,
            'address_space_efficiency': self.effective_capacity / (2**self.address_qubits)
        }

    def memory_usage(self) -> float:
        """Get current memory usage ratio."""
        return self.stored_count / self.effective_capacity

    def reset(self) -> None:
        """Reset memory contents."""
        self.qram.memory.fill(0)
        self.qram._memory_usage = 0.0
        self.stored_count = 0
        self._key_to_address.clear()

    def defragment(self) -> None:
        """
        Defragment memory by compacting stored entries.

        This operation reorganizes memory to improve access patterns.
        """
        # Collect all non-empty entries
        active_entries = []
        for addr in range(self.effective_capacity):
            if np.any(np.abs(self.qram.memory[addr]) > 1e-10):
                active_entries.append((addr, self.qram.memory[addr].copy()))

        # Clear memory
        self.reset()

        # Rewrite entries in compact form
        for i, (_, data) in enumerate(active_entries):
            if i < self.effective_capacity:
                self.qram.memory[i] = data
                self.stored_count += 1

        # Update key mappings (simplified - in practice would need to track keys)
        self._key_to_address = {i: i for i in range(len(active_entries))}

    def validate_quantum_constraints(self) -> Dict[str, bool]:
        """
        Validate that current configuration meets quantum hardware constraints.

        Returns:
            Dictionary of constraint validation results
        """
        hardware_reqs = self.qram.get_hardware_requirements()

        return {
            'total_qubits_feasible': hardware_reqs['total_qubits'] <= 100,  # Current hardware limit
            'circuit_depth_reasonable': hardware_reqs['circuit_depth'] <= 1000,
            'gate_count_manageable': hardware_reqs['gate_count'] <= 10000,
            'memory_size_practical': self.effective_capacity <= 1024,
            'encoding_efficient': self.qram.use_amplitude_encoding,
            'address_space_utilized': self.memory_usage() > 0.1
        }
