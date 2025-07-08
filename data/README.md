# QMNN Dataset Repository

This directory contains datasets, download scripts, and data validation tools for the QMNN project.

## 📊 Available Datasets

### Standard ML Benchmarks
- **MNIST**: Handwritten digits (28×28 grayscale)
- **CIFAR-10**: Natural images (32×32 RGB, 10 classes)
- **Fashion-MNIST**: Fashion items (28×28 grayscale)

### Quantum-Specific Datasets
- **Synthetic Quantum States**: Generated quantum memory patterns
- **QRAM Benchmarks**: Memory access pattern datasets
- **Noise Characterization**: Quantum device noise profiles

### Research Datasets
- **QM9**: Molecular property prediction (for chemistry applications)
- **Quantum Circuit Datasets**: Pre-compiled quantum circuits
- **Memory Task Datasets**: Algorithmic memory tasks

## 🚀 Quick Start

### Download All Datasets
```bash
# Download standard benchmarks
python download_data.py --datasets mnist cifar10 fashion-mnist

# Download quantum-specific data
python download_data.py --datasets quantum-synthetic qram-bench

# Download everything
python download_data.py --all
```

### Verify Data Integrity
```bash
# Check SHA256 checksums
python verify_data.py

# Detailed validation
python verify_data.py --detailed
```

## 📁 Directory Structure

```
data/
├── README.md                 # This file
├── download_data.py          # Data download script
├── verify_data.py           # Data verification script
├── checksums.sha256         # SHA256 checksums
├── raw/                     # Raw downloaded data
│   ├── mnist/
│   ├── cifar10/
│   └── quantum/
├── processed/               # Preprocessed data
│   ├── mnist_sequences.npz
│   ├── quantum_states.npz
│   └── memory_tasks.npz
├── synthetic/               # Generated synthetic data
│   ├── quantum_memory_patterns.npz
│   └── noise_models.json
└── external/                # External dataset links
    ├── qm9_link.txt
    └── quantum_datasets.yaml
```

## 📥 Data Download Scripts

### Basic Usage
```python
from data.download_data import download_dataset

# Download MNIST
download_dataset('mnist', target_dir='data/raw/mnist')

# Download with verification
download_dataset('cifar10', verify_checksum=True)
```

### Command Line Interface
```bash
# Download specific dataset
python download_data.py --dataset mnist --output data/raw/

# Download with progress bar
python download_data.py --dataset cifar10 --verbose

# Resume interrupted download
python download_data.py --dataset qm9 --resume
```

## 🔍 Data Verification

### Checksum Verification
All datasets include SHA256 checksums for integrity verification:

```bash
# Verify all datasets
python verify_data.py

# Verify specific dataset
python verify_data.py --dataset mnist

# Generate new checksums
python verify_data.py --generate-checksums
```

### Data Quality Checks
```bash
# Check data format and structure
python verify_data.py --check-format

# Validate quantum state normalization
python verify_data.py --check-quantum-states

# Check for corrupted files
python verify_data.py --check-corruption
```

## 🧪 Synthetic Data Generation

### Quantum Memory Patterns
```python
from data.synthetic import generate_quantum_memory_patterns

# Generate training patterns
patterns = generate_quantum_memory_patterns(
    n_patterns=1000,
    n_qubits=8,
    pattern_type='random'
)
```

### Memory Task Datasets
```python
from data.synthetic import generate_memory_tasks

# Generate copy task data
copy_data = generate_memory_tasks(
    task_type='copy',
    sequence_lengths=[10, 20, 50],
    n_samples=1000
)
```

## 📊 Dataset Specifications

### MNIST
- **Size**: 70,000 samples (60k train, 10k test)
- **Format**: 28×28 grayscale images
- **Classes**: 10 digits (0-9)
- **File size**: ~11 MB compressed
- **Checksum**: `a684c7c5...` (see checksums.sha256)

### CIFAR-10
- **Size**: 60,000 samples (50k train, 10k test)
- **Format**: 32×32 RGB images
- **Classes**: 10 categories
- **File size**: ~163 MB compressed
- **Checksum**: `c58f30108...` (see checksums.sha256)

### Quantum Synthetic
- **Size**: Variable (configurable)
- **Format**: Complex numpy arrays
- **Qubits**: 4-12 qubits supported
- **States**: Normalized quantum states
- **Generation**: Reproducible with fixed seeds

## 🔧 Data Preprocessing

### Sequence Conversion
Convert image datasets to sequences for RNN training:

```python
from data.preprocessing import images_to_sequences

# Convert MNIST to sequences
sequences = images_to_sequences(
    mnist_data, 
    sequence_length=20,
    overlap=0.1
)
```

### Quantum State Encoding
Encode classical data as quantum states:

```python
from data.preprocessing import classical_to_quantum

# Encode classical vectors as quantum amplitudes
quantum_data = classical_to_quantum(
    classical_data,
    encoding='amplitude',
    n_qubits=8
)
```

## 📈 Data Statistics

### Dataset Sizes
| Dataset | Samples | Size (MB) | Download Time* |
|---------|---------|-----------|----------------|
| MNIST | 70,000 | 11 | ~30s |
| CIFAR-10 | 60,000 | 163 | ~2min |
| Fashion-MNIST | 70,000 | 29 | ~1min |
| QM9 | 134,000 | 2,800 | ~15min |

*Approximate times on 100 Mbps connection

### Storage Requirements
- **Minimum**: 200 MB (basic datasets)
- **Recommended**: 5 GB (all datasets + processed)
- **Full**: 20 GB (including external datasets)

## 🌐 External Datasets

Some large datasets are hosted externally:

### QM9 Molecular Dataset
- **Source**: [Quantum Machine](http://quantum-machine.org/)
- **Size**: ~2.8 GB
- **Access**: Automatic download with agreement
- **Usage**: Molecular property prediction

### IBM Quantum Device Data
- **Source**: IBM Quantum Network
- **Access**: Requires IBM Quantum account
- **Content**: Device calibration and noise data
- **Usage**: Realistic noise modeling

## 🔒 Data Privacy and Licensing

### Licensing
- **MNIST/CIFAR-10**: Public domain / Creative Commons
- **Synthetic Data**: CC0 (public domain)
- **QM9**: Academic use only
- **IBM Data**: Subject to IBM Quantum terms

### Privacy
- No personal data is collected or stored
- All datasets are publicly available or synthetically generated
- Quantum device data is anonymized

## 🛠️ Troubleshooting

### Common Issues

**Download Fails**
```bash
# Check internet connection
ping google.com

# Retry with resume
python download_data.py --dataset mnist --resume

# Use alternative mirror
python download_data.py --dataset mnist --mirror backup
```

**Checksum Mismatch**
```bash
# Re-download corrupted file
python download_data.py --dataset mnist --force-redownload

# Verify specific file
python verify_data.py --file data/raw/mnist/train-images.gz
```

**Insufficient Space**
```bash
# Check available space
df -h

# Clean old downloads
python download_data.py --clean-cache

# Download to external drive
python download_data.py --dataset qm9 --output /external/drive/data/
```

### Getting Help

1. **Check logs**: `data/download.log`
2. **Verify checksums**: `python verify_data.py`
3. **Clean and retry**: `python download_data.py --clean --retry`
4. **Report issues**: Create GitHub issue with error logs

## 📚 References

1. LeCun, Y., et al. "MNIST handwritten digit database." (1998)
2. Krizhevsky, A. "Learning multiple layers of features from tiny images." (2009)
3. Ramakrishnan, R., et al. "Quantum chemistry structures and properties of 134 kilo molecules." Scientific Data 1 (2014)

## 🤝 Contributing

To add new datasets:

1. Add download script to `download_data.py`
2. Include checksum in `checksums.sha256`
3. Add documentation to this README
4. Create verification tests in `verify_data.py`
5. Update preprocessing scripts if needed

---

**Last Updated**: July 2025  
**Maintainer**: QMNN Team  
**Contact**: data@qmnn-project.org
