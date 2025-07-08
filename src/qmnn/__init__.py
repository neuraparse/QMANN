"""
QMNN: Quantum Memory-Augmented Neural Networks

A quantum computing framework for memory-augmented neural networks using QRAM.
Includes 2025 state-of-the-art quantum transformers, fault-tolerant error correction,
and quantum federated learning capabilities.
"""

__version__ = "1.0.0"
__author__ = "Bayram Eker, Neura Parse Research Team"
__email__ = "info@neuraparse.com"
__license__ = "MIT"
__organization__ = "Neura Parse"
__website__ = "https://neuraparse.com"

# Core imports
from .core import QRAM, QuantumMemory
from .models import QMNN, QuantumNeuralNetwork
from .training import QMNNTrainer
from .utils import quantum_state_to_classical, classical_to_quantum_state

# 2025 Advanced Features
try:
    from .quantum_transformers import (
        QuantumAttentionMechanism,
        QuantumTransformerBlock,
        QuantumVisionTransformer,
        QuantumMixedStateAttention
    )
    QUANTUM_TRANSFORMERS_AVAILABLE = True
except ImportError:
    QUANTUM_TRANSFORMERS_AVAILABLE = False

try:
    from .error_correction import (
        SurfaceCodeQRAM,
        AdaptiveErrorCorrection,
        QuantumErrorMitigation
    )
    ERROR_CORRECTION_AVAILABLE = True
except ImportError:
    ERROR_CORRECTION_AVAILABLE = False

try:
    from .quantum_federated import (
        QuantumSecureAggregation,
        QuantumDifferentialPrivacy,
        QuantumFederatedQMNN
    )
    QUANTUM_FEDERATED_AVAILABLE = True
except ImportError:
    QUANTUM_FEDERATED_AVAILABLE = False

try:
    from .quantum_advantage import QuantumAdvantageMetrics
    QUANTUM_ADVANTAGE_AVAILABLE = True
except ImportError:
    QUANTUM_ADVANTAGE_AVAILABLE = False

# Base exports
__all__ = [
    "QRAM",
    "QuantumMemory",
    "QMNN",
    "QuantumNeuralNetwork",
    "QMNNTrainer",
    "quantum_state_to_classical",
    "classical_to_quantum_state",
]

# Add 2025 features if available
if QUANTUM_TRANSFORMERS_AVAILABLE:
    __all__.extend([
        "QuantumAttentionMechanism",
        "QuantumTransformerBlock",
        "QuantumVisionTransformer",
        "QuantumMixedStateAttention",
    ])

if ERROR_CORRECTION_AVAILABLE:
    __all__.extend([
        "SurfaceCodeQRAM",
        "AdaptiveErrorCorrection",
        "QuantumErrorMitigation",
    ])

if QUANTUM_FEDERATED_AVAILABLE:
    __all__.extend([
        "QuantumSecureAggregation",
        "QuantumDifferentialPrivacy",
        "QuantumFederatedQMNN",
    ])

if QUANTUM_ADVANTAGE_AVAILABLE:
    __all__.extend([
        "QuantumAdvantageMetrics",
    ])

# Version info for 2025 features
FEATURES_2025 = {
    "quantum_transformers": QUANTUM_TRANSFORMERS_AVAILABLE,
    "fault_tolerant_qec": ERROR_CORRECTION_AVAILABLE,
    "quantum_federated": QUANTUM_FEDERATED_AVAILABLE,
    "advantage_verification": QUANTUM_ADVANTAGE_AVAILABLE,
}
