"""
QMANN: Quantum Memory-Augmented Neural Networks

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
from .models import QMANN, QuantumNeuralNetwork
from .training import QMANNTrainer
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
        QuantumFederatedQMANN
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
        ExperimentalQMANN,
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
    "QMANN",
    "QuantumNeuralNetwork",
    "QMANNTrainer",
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
        "QuantumFederatedQMANN",
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
        "ExperimentalQMANN",
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

# Add mode selection helpers
__all__.extend([
    "recommend_mode",
    "print_mode_guide",
    "quick_start"
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
        'qmann_version': __version__,
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

def recommend_mode(purpose: str = None, budget_usd: float = 0.0, n_qubits: int = None) -> str:
    """
    Recommend the best QMANN mode based on user requirements.

    Args:
        purpose: Purpose of experiment ('research', 'development', 'validation', 'production')
        budget_usd: Available budget in USD (0 = free only)
        n_qubits: Required number of qubits

    Returns:
        Recommended mode with explanation
    """
    recommendations = []

    # Budget-based recommendations
    if budget_usd == 0.0:
        if purpose in ['research', 'theory', 'paper']:
            recommendations.append("🔬 THEORETICAL MODE - Perfect for research papers and theoretical analysis")
        else:
            recommendations.append("💻 SIMULATION MODE - Free classical simulation for development")
    else:
        recommendations.append("⚛️ HARDWARE MODE - Real quantum hardware experiments (costs money)")

    # Qubit-based recommendations
    if n_qubits is not None:
        if n_qubits > 20:
            recommendations.append("🔬 THEORETICAL MODE - Only theoretical mode supports >20 qubits")
        elif n_qubits > 12:
            recommendations.append("💻 SIMULATION MODE - Hardware limited to 12 qubits currently")
        else:
            if budget_usd > 0:
                recommendations.append("⚛️ HARDWARE MODE - Suitable for real quantum hardware")
            else:
                recommendations.append("💻 SIMULATION MODE - Free simulation available")

    # Purpose-based recommendations
    purpose_map = {
        'research': "🔬 THEORETICAL MODE",
        'theory': "🔬 THEORETICAL MODE",
        'paper': "🔬 THEORETICAL MODE",
        'development': "💻 SIMULATION MODE",
        'testing': "💻 SIMULATION MODE",
        'education': "💻 SIMULATION MODE",
        'validation': "⚛️ HARDWARE MODE (if budget allows)",
        'production': "⚛️ HARDWARE MODE",
        'benchmark': "⚛️ HARDWARE MODE"
    }

    if purpose and purpose.lower() in purpose_map:
        recommendations.append(f"Based on purpose '{purpose}': {purpose_map[purpose.lower()]}")

    # Default recommendation
    if not recommendations:
        recommendations.append("💻 SIMULATION MODE - Good default for most users (free)")

    return "\n".join(recommendations)

def print_mode_guide():
    """Print comprehensive mode selection guide."""
    print("🎮 QMANN MODE SELECTION GUIDE")
    print("=" * 50)
    print()

    print("🔬 THEORETICAL MODE (FREE)")
    print("  Purpose: Research papers, theoretical analysis, algorithm design")
    print("  Resources: Unlimited qubits, perfect gates, infinite coherence")
    print("  Example: python examples/01_theoretical_mode.py")
    print()

    print("💻 SIMULATION MODE (FREE)")
    print("  Purpose: Algorithm development, testing, education")
    print("  Resources: Up to 20 qubits, noise modeling, classical simulation")
    print("  Example: python examples/02_simulation_mode.py")
    print()

    print("⚛️ HARDWARE MODE (PAID)")
    print("  Purpose: Real quantum hardware validation, proof-of-concept")
    print("  Resources: 4-12 qubits, real noise, actual quantum effects")
    print("  Cost: IBM ~$0.001/shot, IonQ ~$0.01/shot")
    print("  Example: python examples/03_hardware_mode.py")
    print()

    print("💡 QUICK RECOMMENDATIONS:")
    print("  - Writing a paper? → Theoretical Mode")
    print("  - Learning QMANN? → Simulation Mode")
    print("  - Testing algorithms? → Simulation Mode")
    print("  - Need real quantum results? → Hardware Mode (estimate costs first!)")
    print()

    print("💰 COST ESTIMATION:")
    print("  python scripts/estimate_hardware_costs.py --qubits 6 --shots 1000")
    print()

    # Show available features
    features = get_available_features()
    print("✅ AVAILABLE FEATURES:")
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")

def quick_start(mode: str = "simulation"):
    """
    Quick start function for different modes.

    Args:
        mode: Mode to start with ('theoretical', 'simulation', 'hardware')
    """
    mode = mode.lower()

    if mode == "theoretical":
        print("🔬 Starting QMANN in Theoretical Mode...")
        if CONFIG_AVAILABLE:
            from .config import THEORETICAL_ANALYSIS, validate_experimental_setup
            validate_experimental_setup(THEORETICAL_ANALYSIS)
            print("✅ Theoretical configuration validated")
        print("📖 Run: python examples/01_theoretical_mode.py")

    elif mode == "simulation":
        print("💻 Starting QMANN in Simulation Mode...")
        if CONFIG_AVAILABLE:
            from .config import SIMULATION_VALIDATION, validate_experimental_setup
            validate_experimental_setup(SIMULATION_VALIDATION)
            print("✅ Simulation configuration validated")
        print("📖 Run: python examples/02_simulation_mode.py")

    elif mode == "hardware":
        print("⚛️ Starting QMANN in Hardware Mode...")
        print("⚠️  WARNING: This mode costs real money!")
        if CONFIG_AVAILABLE:
            from .config import HARDWARE_PROOF_OF_CONCEPT, validate_experimental_setup
            validate_experimental_setup(HARDWARE_PROOF_OF_CONCEPT)
            print("✅ Hardware configuration validated")
        print("💰 Estimate costs first: python scripts/estimate_hardware_costs.py")
        print("📖 Run: python examples/03_hardware_mode.py")

    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: 'theoretical', 'simulation', 'hardware'")
        print_mode_guide()
