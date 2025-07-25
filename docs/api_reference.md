# QMNN API Reference

Complete API reference for Quantum Memory-Augmented Neural Networks.

## Core Components

### QRAM (Quantum Random Access Memory)

```python
class QRAM:
    """Realistic Quantum Random Access Memory implementation."""
    
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
```

**Methods:**
- `write(address: int, data: np.ndarray) -> None`: Write data to memory
- `read(address: int) -> np.ndarray`: Read data from memory
- `superposition_read(query: np.ndarray) -> np.ndarray`: Quantum superposition read
- `get_memory_info() -> Dict`: Get memory statistics
- `validate_quantum_constraints() -> Dict[str, bool]`: Check hardware constraints

### QuantumMemory

```python
class QuantumMemory:
    """High-level quantum memory with associative access."""
    
    def __init__(self, capacity: int, embedding_dim: int, 
                 max_qubits: int = 16, use_amplitude_encoding: bool = True):
        """
        Initialize quantum memory.
        
        Args:
            capacity: Memory capacity (auto-adjusted for hardware)
            embedding_dim: Dimension of stored embeddings
            max_qubits: Maximum qubits available
            use_amplitude_encoding: Use amplitude vs basis encoding
        """
```

**Methods:**
- `store_embedding(key: np.ndarray, value: np.ndarray) -> int`: Store key-value pair
- `retrieve_embedding(query: np.ndarray) -> np.ndarray`: Retrieve by similarity
- `memory_usage() -> float`: Get current memory utilization
- `capacity_bound() -> Dict`: Get capacity information

### QMNN (Quantum Memory-Augmented Neural Network)

```python
class QMNN(nn.Module):
    """Main QMNN model combining classical and quantum components."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 memory_capacity: int = 128, memory_embedding_dim: int = 64,
                 n_quantum_layers: int = 2, max_qubits: int = 16,
                 use_attention: bool = True, dropout: float = 0.1,
                 memory_regularizer: float = 0.01):
        """
        Initialize QMNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            memory_capacity: Quantum memory capacity
            memory_embedding_dim: Memory embedding dimension
            n_quantum_layers: Number of quantum processing layers
            max_qubits: Maximum qubits for quantum operations
            use_attention: Whether to use attention mechanism
            dropout: Dropout rate
            memory_regularizer: Memory usage regularization
        """
```

**Methods:**
- `forward(x: torch.Tensor) -> Tuple[torch.Tensor, Dict]`: Forward pass
- `quantum_memory_read(query: torch.Tensor) -> torch.Tensor`: Read from quantum memory
- `quantum_memory_write(key: torch.Tensor, value: torch.Tensor) -> int`: Write to memory
- `get_quantum_info() -> Dict`: Get quantum component information

## Quantum Transformers

### QuantumTransformer

```python
class QuantumTransformer(nn.Module):
    """Practical Quantum Transformer for sequence modeling."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1, n_qubits: int = 8, 
                 use_full_quantum: bool = False):
```

**Methods:**
- `forward(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor`
- `generate(input_ids: torch.Tensor, max_length: int, temperature: float) -> torch.Tensor`
- `get_quantum_info() -> Dict[str, Any]`: Get quantum configuration info

### QuantumAttentionMechanism

```python
class QuantumAttentionMechanism(nn.Module):
    """Optimized quantum-enhanced attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int = 8, n_qubits: int = 8,
                 use_full_quantum: bool = False):
```

**Methods:**
- `forward(query, key, value, mask=None) -> Tuple[torch.Tensor, torch.Tensor]`
- `compute_quantum_inspired_attention(q, k) -> torch.Tensor`
- `get_quantum_info() -> Dict[str, Any]`

## Error Correction

### PracticalErrorCorrection

```python
class PracticalErrorCorrection(nn.Module):
    """Practical quantum error correction for near-term devices."""
    
    def __init__(self, n_qubits: int, error_rate: float = 0.01,
                 correction_method: str = "repetition", code_distance: int = 3):
        """
        Args:
            n_qubits: Number of logical qubits
            error_rate: Expected error rate
            correction_method: "repetition", "surface", or "steane"
            code_distance: Error correction code distance
        """
```

**Methods:**
- `detect_errors(quantum_state: torch.Tensor) -> Dict[str, torch.Tensor]`
- `correct_errors(quantum_state: torch.Tensor, error_info: Dict) -> torch.Tensor`
- `create_error_correction_circuit(logical_qubits: int) -> QuantumCircuit`

## Federated Learning

### QuantumFederatedQMNN

```python
class QuantumFederatedQMNN(nn.Module):
    """Quantum Federated QMNN System."""
    
    def __init__(self, base_model_config: Dict[str, Any],
                 n_clients: int = 10, privacy_epsilon: float = 1.0):
```

**Methods:**
- `federated_round(client_data: List[Tuple], local_epochs: int) -> Dict[str, float]`
- `client_update(client_id: int, local_data, local_targets) -> Dict[str, torch.Tensor]`
- `aggregate_updates(client_updates: List[Dict]) -> Dict[str, torch.Tensor]`

### QuantumSecureAggregation

```python
class QuantumSecureAggregation(nn.Module):
    """Quantum Secure Aggregation Protocol."""
    
    def __init__(self, n_clients: int, n_qubits: int = 8,
                 security_parameter: int = 128):
```

**Methods:**
- `secure_aggregate(client_updates: List[torch.Tensor]) -> torch.Tensor`
- `quantum_secret_share(secret: torch.Tensor) -> List[torch.Tensor]`
- `quantum_secret_reconstruct(shares: List[torch.Tensor]) -> torch.Tensor`

## Modular Components

### Quantum Memory Management

```python
class QuantumMemoryManager:
    """High-level manager for quantum memory operations."""
    
    def __init__(self, memory_configs: List[Dict]):
```

**Methods:**
- `allocate_memory(size: int, memory_type: str) -> int`
- `get_memory_stats() -> Dict[str, Union[int, float, Dict]]`
- `optimize_memory_layout() -> None`
- `validate_quantum_constraints() -> Dict[str, bool]`

### Classical Controllers

```python
class HybridController(nn.Module):
    """Hybrid controller managing classical and quantum components."""
    
    def __init__(self, config: Dict[str, Any]):
```

**Methods:**
- `forward(x: torch.Tensor, quantum_processor=None) -> Dict[str, torch.Tensor]`
- `get_controller_info() -> Dict[str, Any]`

### Attention Mechanisms

```python
class AdaptiveAttention(nn.Module):
    """Adaptive attention switching between classical and quantum modes."""
    
    def __init__(self, d_model: int, n_heads: int = 8, n_qubits: int = 8,
                 dropout: float = 0.1, adaptation_threshold: float = 0.5):
```

### Hybrid Layers

```python
class HybridTransformerLayer(nn.Module):
    """Hybrid transformer layer combining classical and quantum attention."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1, attention_type: str = "adaptive",
                 n_qubits: int = 8):
```

## Training Utilities

### QMNNTrainer

```python
class QMNNTrainer:
    """Trainer for QMNN models with quantum-aware optimizations."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3,
                 device: str = 'cpu', use_wandb: bool = False):
```

**Methods:**
- `train_epoch(X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]`
- `validate(X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]`
- `save_checkpoint(path: str) -> None`
- `load_checkpoint(path: str) -> None`

## Utility Functions

### System Information

```python
def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""

def get_available_features() -> Dict[str, bool]:
    """Get information about available features and modules."""
```

### Quantum Metrics

```python
def quantum_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """Compute quantum state fidelity."""

def quantum_advantage_metric(quantum_model, classical_model, test_data) -> Dict:
    """Compute quantum advantage metrics."""

def memory_efficiency(model: QMNN) -> Dict[str, float]:
    """Compute memory efficiency metrics."""
```

## Hardware Requirements

### Recommended Configurations

| Use Case | Qubits | Memory Capacity | RAM | GPU Memory |
|----------|--------|-----------------|-----|------------|
| Research | 6-8 | 32-64 | 8GB | 4GB |
| Development | 8-10 | 64-128 | 16GB | 8GB |
| Production | 10-12 | 128-512 | 32GB | 16GB |

### Constraints and Limitations

- **Maximum Qubits**: 12 (current hardware limit)
- **Memory Capacity**: Auto-adjusted based on available qubits
- **Sequence Length**: Limited by memory capacity
- **Batch Size**: Recommended 16-64 for quantum components

## Error Handling

### Common Exceptions

```python
class QuantumMemoryError(Exception):
    """Raised when quantum memory operations fail."""

class QubitLimitExceeded(Exception):
    """Raised when qubit requirements exceed hardware limits."""

class QuantumCircuitError(Exception):
    """Raised when quantum circuit construction fails."""
```

### Warning Categories

- `QuantumHardwareWarning`: Hardware constraint warnings
- `MemoryCapacityWarning`: Memory usage warnings
- `PerformanceWarning`: Performance optimization suggestions
