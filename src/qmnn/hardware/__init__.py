"""
Quantum Hardware Interface Module for QMANN

This module provides interfaces to real quantum hardware for experimental
validation of QMANN algorithms on NISQ devices.
"""

from .quantum_backend import (
    QuantumBackend,
    IBMQuantumBackend,
    GoogleQuantumBackend,
    IonQBackend,
    QuantumBackendManager
)

try:
    from .experimental_interface import (
        ExperimentalQMANN,
        HardwareAwareQRAM,
        NISQOptimizedLayers
    )
    EXPERIMENTAL_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_AVAILABLE = False

__all__ = [
    'QuantumBackend',
    'IBMQuantumBackend', 
    'GoogleQuantumBackend',
    'IonQBackend',
    'QuantumBackendManager'
]

if EXPERIMENTAL_AVAILABLE:
    __all__.extend([
        'ExperimentalQMANN',
        'HardwareAwareQRAM', 
        'NISQOptimizedLayers'
    ])
