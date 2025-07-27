"""
Configuration Module for QMANN

This module provides configuration management for different experimental modes:
- Theoretical analysis (ideal quantum computers)
- Classical simulation (quantum-inspired algorithms)
- Real hardware experiments (NISQ devices)
"""

from .experimental_config import (
    ExperimentMode,
    QuantumBackendType,
    TheoreticalConfig,
    SimulationConfig,
    HardwareConfig,
    ExperimentalConfig,
    THEORETICAL_ANALYSIS,
    SIMULATION_VALIDATION,
    HARDWARE_PROOF_OF_CONCEPT,
    HARDWARE_FULL_EXPERIMENT,
    get_recommended_config,
    validate_experimental_setup
)

__all__ = [
    'ExperimentMode',
    'QuantumBackendType', 
    'TheoreticalConfig',
    'SimulationConfig',
    'HardwareConfig',
    'ExperimentalConfig',
    'THEORETICAL_ANALYSIS',
    'SIMULATION_VALIDATION', 
    'HARDWARE_PROOF_OF_CONCEPT',
    'HARDWARE_FULL_EXPERIMENT',
    'get_recommended_config',
    'validate_experimental_setup'
]
