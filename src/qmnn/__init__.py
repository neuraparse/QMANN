"""
QMNN: Quantum Memory-Augmented Neural Networks

A quantum computing framework for memory-augmented neural networks using QRAM.
Includes 2025 state-of-the-art quantum transformers, fault-tolerant error correction,
and quantum federated learning capabilities.
"""

__version__ = "1.0.0"
__author__ = "Bayram Eker, Neura Parse Research Team"
__email__ = "info@neuraparse.com"
__license__ = "Apache-2.0"
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

# Modular Components (2025)
try:
    from .quantum_memory import (
        QuantumMemoryManager,
        AdaptiveQuantumMemory
    )
    QUANTUM_MEMORY_AVAILABLE = True
except ImportError:
    QUANTUM_MEMORY_AVAILABLE = False

try:
    from .classical_controller import (
        ClassicalEncoder,
        LSTMController,
        ClassicalDecoder,
        HybridController
    )
    CLASSICAL_CONTROLLER_AVAILABLE = True
except ImportError:
    CLASSICAL_CONTROLLER_AVAILABLE = False

try:
    from .attention import (
        MultiHeadAttention,
        QuantumInspiredAttention,
        AdaptiveAttention
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False

try:
    from .hybrid_layers import (
        HybridTransformerLayer,
        QuantumClassicalFusion,
        AdaptiveHybridLayer,
        HybridResidualBlock
    )
    HYBRID_LAYERS_AVAILABLE = True
except ImportError:
    HYBRID_LAYERS_AVAILABLE = False

# Real Quantum Hardware Interface (2025)
try:
    from .hardware import (
        QuantumBackendManager,
        IBMQuantumBackend,
        GoogleQuantumBackend,
        IonQBackend,
        ExperimentalQMNN,
        HardwareAwareQRAM,
        NISQOptimizedLayers
    )
    QUANTUM_HARDWARE_AVAILABLE = True
except ImportError:
    QUANTUM_HARDWARE_AVAILABLE = False

# Experimental Configuration (2025)
try:
    from .config import (
        ExperimentalConfig,
        ExperimentMode,
        THEORETICAL_ANALYSIS,
        SIMULATION_VALIDATION,
        HARDWARE_PROOF_OF_CONCEPT,
        get_recommended_config,
        validate_experimental_setup
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

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

# Add modular components if available
if QUANTUM_MEMORY_AVAILABLE:
    __all__.extend([
        "QuantumMemoryManager",
        "AdaptiveQuantumMemory",
    ])

if CLASSICAL_CONTROLLER_AVAILABLE:
    __all__.extend([
        "ClassicalEncoder",
        "LSTMController",
        "ClassicalDecoder",
        "HybridController",
    ])

if ATTENTION_AVAILABLE:
    __all__.extend([
        "MultiHeadAttention",
        "QuantumInspiredAttention",
        "AdaptiveAttention",
    ])

if HYBRID_LAYERS_AVAILABLE:
    __all__.extend([
        "HybridTransformerLayer",
        "QuantumClassicalFusion",
        "AdaptiveHybridLayer",
        "HybridResidualBlock",
    ])

# Add quantum hardware components if available
if QUANTUM_HARDWARE_AVAILABLE:
    __all__.extend([
        "QuantumBackendManager",
        "IBMQuantumBackend",
        "GoogleQuantumBackend",
        "IonQBackend",
        "ExperimentalQMNN",
        "HardwareAwareQRAM",
        "NISQOptimizedLayers",
    ])

# Add experimental configuration if available
if CONFIG_AVAILABLE:
    __all__.extend([
        "ExperimentalConfig",
        "ExperimentMode",
        "THEORETICAL_ANALYSIS",
        "SIMULATION_VALIDATION",
        "HARDWARE_PROOF_OF_CONCEPT",
        "get_recommended_config",
        "validate_experimental_setup",
    ])

# Version info for 2025 features
FEATURES_2025 = {
    "quantum_transformers": QUANTUM_TRANSFORMERS_AVAILABLE,
    "fault_tolerant_qec": ERROR_CORRECTION_AVAILABLE,
    "quantum_federated": QUANTUM_FEDERATED_AVAILABLE,
    "advantage_verification": QUANTUM_ADVANTAGE_AVAILABLE,
    "modular_quantum_memory": QUANTUM_MEMORY_AVAILABLE,
    "classical_controllers": CLASSICAL_CONTROLLER_AVAILABLE,
    "advanced_attention": ATTENTION_AVAILABLE,
    "hybrid_layers": HYBRID_LAYERS_AVAILABLE,
    "real_quantum_hardware": QUANTUM_HARDWARE_AVAILABLE,
    "experimental_configuration": CONFIG_AVAILABLE,
}

def get_available_features():
    """Get information about available features and modules."""
    return FEATURES_2025

def get_system_info():
    """Get comprehensive system information."""
    import torch

    info = {
        'qmnn_version': __version__,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'available_features': get_available_features(),
        'recommended_config': {
            'max_qubits': 12,
            'max_memory_capacity': 512,
            'recommended_batch_size': 32,
            'hardware_requirements': {
                'min_ram_gb': 8,
                'recommended_ram_gb': 16,
                'gpu_memory_gb': 4
            }
        }
    }

    return info
