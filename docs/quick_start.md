# QMANN Quick Start Guide

Welcome to Quantum Memory-Augmented Neural Networks (QMANN)! This guide will help you get started with QMANN in just a few minutes.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Qiskit 0.45+
- PennyLane 0.32+

### Install QMANN

```bash
# Clone the repository
git clone https://github.com/neuraparse/QMANN.git
cd QMANN

# Install dependencies
pip install -r requirements.txt

# Install QMANN in development mode
pip install -e .

# Quick test
make quicktest
```

### Docker Installation (Recommended)

```bash
# Build and run with Docker
make docker-build
make docker-run

# Or use pre-built image
docker run -it qmann:latest python -c "import qmann; print('QMANN ready!')"
```

## Your First QMANN Model

### Basic Example: Sequential Classification

```python
import torch
import numpy as np
from qmann import QMANN, QMANNTrainer

# Create sample data
batch_size, seq_len, input_dim = 32, 10, 8
X = torch.randn(batch_size, seq_len, input_dim)
y = torch.randint(0, 3, (batch_size, seq_len))

# Initialize QMANN model
model = QMANN(
    input_dim=input_dim,
    hidden_dim=64,
    output_dim=3,  # 3 classes
    memory_capacity=32,
    memory_embedding_dim=16,
    max_qubits=8  # Hardware constraint
)

# Create trainer
trainer = QMANNTrainer(
    model=model,
    learning_rate=1e-3,
    device='cpu'  # Use 'cuda' if available
)

# Train the model
trainer.train_epoch(X, y)

# Make predictions
with torch.no_grad():
    predictions, memory_info = model(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Memory usage: {memory_info['memory_usage']:.2f}")
```

### Memory Operations Example

```python
from qmann import QuantumMemory

# Create quantum memory
memory = QuantumMemory(
    capacity=64,
    embedding_dim=32,
    max_qubits=8
)

# Store and retrieve data
key = np.random.randn(32)
value = np.random.randn(32)

# Store in memory
address = memory.store_embedding(key, value)
print(f"Stored at address: {address}")

# Retrieve from memory
query = key + 0.1 * np.random.randn(32)  # Noisy query
retrieved = memory.retrieve_embedding(query)
print(f"Retrieval similarity: {np.dot(value, retrieved):.3f}")

# Memory statistics
stats = memory.get_memory_info()
print(f"Memory utilization: {stats['capacity_utilization']:.2f}")
```

## Advanced Features

### Quantum Transformers

```python
from qmann import QuantumTransformer

# Create quantum transformer for text processing
model = QuantumTransformer(
    vocab_size=1000,
    d_model=256,
    n_layers=4,
    n_heads=8,
    n_qubits=8,
    use_full_quantum=False  # Use efficient quantum-inspired mode
)

# Generate text
input_ids = torch.randint(0, 1000, (1, 10))
generated = model.generate(input_ids, max_length=20)
print(f"Generated sequence: {generated}")
```

### Quantum Federated Learning

```python
from qmann import QuantumFederatedQMANN

# Create federated learning setup
fed_model = QuantumFederatedQMANN(
    base_model_config={
        'input_dim': 8,
        'hidden_dim': 32,
        'output_dim': 3,
        'memory_capacity': 16
    },
    n_clients=5,
    privacy_epsilon=1.0
)

# Simulate federated training round
client_data = [
    (torch.randn(16, 5, 8), torch.randint(0, 3, (16, 5)))
    for _ in range(5)
]

metrics = fed_model.federated_round(client_data)
print(f"Round {metrics['round']}: avg loss = {metrics['average_client_loss']:.3f}")
```

### Error Correction

```python
from qmann import PracticalErrorCorrection

# Create error correction system
error_corrector = PracticalErrorCorrection(
    n_qubits=8,
    error_rate=0.01,
    correction_method="repetition",  # or "surface", "steane"
    code_distance=3
)

# Simulate quantum state with errors
quantum_state = torch.randn(32, 8)  # Batch of quantum states

# Detect and correct errors
error_info = error_corrector.detect_errors(quantum_state)
corrected_state = error_corrector.correct_errors(quantum_state, error_info)

print(f"Errors detected: {error_info['error_detected'].sum().item()}")
print(f"Correction success rate: {error_corrector.correction_success_rate:.3f}")
```

## Hardware Considerations

### Quantum Hardware Limits

QMANN is designed to work within realistic quantum hardware constraints:

```python
# Check system capabilities
from qmann import get_system_info

info = get_system_info()
print("System Information:")
print(f"- Max recommended qubits: {info['recommended_config']['max_qubits']}")
print(f"- Max memory capacity: {info['recommended_config']['max_memory_capacity']}")
print(f"- Available features: {list(info['available_features'].keys())}")
```

### Performance Optimization

```python
# Use hardware-aware configuration
model = QMANN(
    input_dim=16,
    hidden_dim=128,
    output_dim=10,
    memory_capacity=64,  # Keep reasonable
    max_qubits=8,        # Hardware limit
    n_quantum_layers=2,  # Don't overdo quantum layers
    use_attention=True,
    memory_regularizer=0.01  # Prevent memory overflow
)

# Monitor quantum vs classical performance
quantum_info = model.get_quantum_info()
print(f"Quantum parameters: {quantum_info['quantum_parameters']}")
print(f"Classical parameters: {quantum_info['classical_parameters']}")
print(f"Quantum ratio: {quantum_info['quantum_ratio']:.2%}")
```

## Benchmarking

### Run Standard Benchmarks

```bash
# Run all benchmarks
python benchmarks/run.py --all

# Run specific benchmark
python benchmarks/run.py --task mnist_sequential --model qmann

# Compare with baselines
python benchmarks/run.py --task copy_task --models qmann,lstm,transformer
```

### Custom Benchmark

```python
from qmann.benchmarks import MemoryTaskBenchmark

# Create custom memory task
benchmark = MemoryTaskBenchmark(
    task_type="associative_recall",
    sequence_length=20,
    memory_size=50,
    num_samples=1000
)

# Test QMANN vs baselines
results = benchmark.run_comparison(['qmann', 'lstm', 'ntm'])
print(f"QMANN accuracy: {results['qmann']['accuracy']:.3f}")
print(f"LSTM accuracy: {results['lstm']['accuracy']:.3f}")
```

## Next Steps

1. **Explore Examples**: Check out `examples/` directory for more detailed examples
2. **Read Documentation**: See `docs/` for comprehensive API reference
3. **Run Benchmarks**: Use `benchmarks/` to reproduce paper results
4. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

## Common Issues

### Memory Errors
```python
# If you get memory errors, reduce capacity
model = QMANN(memory_capacity=32, max_qubits=6)  # Smaller config
```

### Slow Training
```python
# Use efficient quantum-inspired mode
model = QMANN(use_full_quantum=False)  # Faster training
```

### Hardware Warnings
```python
# Warnings about qubit limits are normal - the system will adapt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qmann")
```

## Support

- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/neuraparse/QMANN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neuraparse/QMANN/discussions)

Happy quantum computing! 🚀
