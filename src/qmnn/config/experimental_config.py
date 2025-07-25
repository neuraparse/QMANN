"""
Experimental Configuration for QMNN

This module separates theoretical and experimental configurations,
ensuring clear distinction between simulation and real hardware experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import warnings
from enum import Enum


class ExperimentMode(Enum):
    """Experiment execution modes."""
    THEORETICAL = "theoretical"  # Pure theoretical analysis
    SIMULATION = "simulation"    # Classical simulation of quantum operations
    HARDWARE = "hardware"        # Real quantum hardware experiments
    HYBRID = "hybrid"           # Mix of simulation and hardware


class QuantumBackendType(Enum):
    """Supported quantum backend types."""
    IBM_QUANTUM = "ibm"
    GOOGLE_QUANTUM = "google"
    IONQ = "ionq"
    AWS_BRAKET = "aws"
    AZURE_QUANTUM = "azure"
    RIGETTI = "rigetti"
    SIMULATOR = "simulator"


@dataclass
class TheoreticalConfig:
    """Configuration for theoretical analysis."""
    
    # Theoretical limits (ideal quantum computer)
    max_qubits: int = 1000  # Theoretical limit
    perfect_gates: bool = True  # No gate errors
    infinite_coherence: bool = True  # No decoherence
    ideal_measurement: bool = True  # Perfect measurement
    
    # Memory capacity (theoretical)
    exponential_memory: bool = True  # 2^n capacity
    superposition_encoding: bool = True  # Full amplitude encoding
    entanglement_advantage: bool = True  # Quantum entanglement benefits
    
    # Complexity advantages
    logarithmic_access: bool = True  # O(log n) access time
    parallel_processing: bool = True  # Quantum parallelism
    
    def validate(self) -> List[str]:
        """Validate theoretical configuration."""
        warnings_list = []
        
        if self.max_qubits > 1000:
            warnings_list.append("max_qubits > 1000 is beyond current theoretical models")
            
        return warnings_list


@dataclass
class SimulationConfig:
    """Configuration for classical simulation of quantum operations."""
    
    # Simulation limits
    max_qubits: int = 20  # Classical simulation limit
    max_circuit_depth: int = 100  # Simulation complexity limit
    shots: int = 1024  # Number of measurement shots
    
    # Noise modeling
    gate_error_rate: float = 0.001  # Simulated gate error rate
    measurement_error_rate: float = 0.01  # Simulated measurement error
    decoherence_time: float = 100.0  # Simulated T2 time (μs)
    
    # Memory simulation
    memory_efficiency: float = 0.7  # Fraction of theoretical capacity
    encoding_efficiency: float = 0.8  # Encoding/decoding efficiency
    
    # Performance
    use_gpu: bool = True  # GPU acceleration for simulation
    parallel_circuits: bool = True  # Parallel circuit execution
    
    def validate(self) -> List[str]:
        """Validate simulation configuration."""
        warnings_list = []
        
        if self.max_qubits > 25:
            warnings_list.append("max_qubits > 25 may be too slow for classical simulation")
            
        if self.shots < 100:
            warnings_list.append("shots < 100 may give unreliable statistics")
            
        if self.gate_error_rate > 0.1:
            warnings_list.append("gate_error_rate > 0.1 is unrealistically high")
            
        return warnings_list


@dataclass
class HardwareConfig:
    """Configuration for real quantum hardware experiments."""
    
    # Hardware constraints (2025 NISQ era)
    max_qubits: int = 12  # Practical limit for current hardware
    max_circuit_depth: int = 50  # NISQ depth limit
    max_two_qubit_gates: int = 25  # Noise accumulation limit
    
    # Backend preferences
    preferred_backends: List[QuantumBackendType] = field(
        default_factory=lambda: [QuantumBackendType.IBM_QUANTUM, QuantumBackendType.SIMULATOR]
    )
    
    # Error mitigation
    error_mitigation: bool = True  # Use error mitigation techniques
    readout_error_mitigation: bool = True  # Correct readout errors
    zero_noise_extrapolation: bool = False  # Advanced error mitigation
    
    # Execution parameters
    shots: int = 1024  # Measurement shots (costs money on real hardware!)
    optimization_level: int = 2  # Circuit optimization (0-3)
    
    # Cost management
    max_cost_per_experiment: float = 10.0  # USD limit per experiment
    use_free_backends_only: bool = True  # Avoid paid backends
    
    # Hardware-specific settings
    ibm_settings: Dict[str, Any] = field(default_factory=lambda: {
        'hub': 'ibm-q',
        'group': 'open',
        'project': 'main',
        'backend_name': 'ibm_brisbane'  # Default IBM backend
    })
    
    google_settings: Dict[str, Any] = field(default_factory=lambda: {
        'processor_id': 'rainbow',
        'project_id': None  # Must be set by user
    })
    
    ionq_settings: Dict[str, Any] = field(default_factory=lambda: {
        'device': 'ionq_qpu',
        'api_key': None  # Must be set by user
    })
    
    def validate(self) -> List[str]:
        """Validate hardware configuration."""
        warnings_list = []
        
        if self.max_qubits > 20:
            warnings_list.append("max_qubits > 20 exceeds current hardware capabilities")
            
        if self.max_circuit_depth > 100:
            warnings_list.append("max_circuit_depth > 100 may fail on NISQ devices")
            
        if self.shots > 10000:
            warnings_list.append("shots > 10000 will be expensive on real hardware")
            
        if not self.use_free_backends_only and self.max_cost_per_experiment < 1.0:
            warnings_list.append("max_cost_per_experiment < $1 may be too low for paid backends")
            
        return warnings_list


@dataclass
class ExperimentalConfig:
    """Complete experimental configuration combining all modes."""
    
    # Experiment mode
    mode: ExperimentMode = ExperimentMode.SIMULATION
    
    # Mode-specific configurations
    theoretical: TheoreticalConfig = field(default_factory=TheoreticalConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # General settings
    experiment_name: str = "qmnn_experiment"
    description: str = "QMNN experimental validation"
    
    # Logging and output
    log_level: str = "INFO"
    save_results: bool = True
    output_dir: str = "experiments/"
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    def get_active_config(self) -> Union[TheoreticalConfig, SimulationConfig, HardwareConfig]:
        """Get the active configuration based on experiment mode."""
        if self.mode == ExperimentMode.THEORETICAL:
            return self.theoretical
        elif self.mode == ExperimentMode.SIMULATION:
            return self.simulation
        elif self.mode == ExperimentMode.HARDWARE:
            return self.hardware
        else:  # HYBRID
            return self.simulation  # Default to simulation for hybrid
            
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configurations."""
        return {
            'theoretical': self.theoretical.validate(),
            'simulation': self.simulation.validate(),
            'hardware': self.hardware.validate()
        }
        
    def get_hardware_requirements(self) -> Dict[str, Any]:
        """Get hardware requirements for current configuration."""
        active_config = self.get_active_config()
        
        if self.mode == ExperimentMode.THEORETICAL:
            return {
                'type': 'theoretical',
                'qubits': 'unlimited',
                'gates': 'perfect',
                'coherence': 'infinite',
                'cost': 'free'
            }
        elif self.mode == ExperimentMode.SIMULATION:
            return {
                'type': 'classical_simulation',
                'qubits': active_config.max_qubits,
                'memory_gb': 2 ** (active_config.max_qubits - 10),  # Rough estimate
                'gpu_recommended': active_config.use_gpu,
                'cost': 'free'
            }
        else:  # HARDWARE
            return {
                'type': 'quantum_hardware',
                'qubits': active_config.max_qubits,
                'backends': [b.value for b in active_config.preferred_backends],
                'estimated_cost_usd': self._estimate_hardware_cost(),
                'api_credentials_required': True
            }
            
    def _estimate_hardware_cost(self) -> float:
        """Estimate cost for hardware experiments."""
        if self.hardware.use_free_backends_only:
            return 0.0
            
        # Rough cost estimates (2025 prices)
        cost_per_shot = {
            QuantumBackendType.IBM_QUANTUM: 0.001,  # $0.001 per shot
            QuantumBackendType.IONQ: 0.01,          # $0.01 per shot
            QuantumBackendType.GOOGLE_QUANTUM: 0.005, # $0.005 per shot
        }
        
        total_cost = 0.0
        for backend in self.hardware.preferred_backends:
            if backend in cost_per_shot:
                total_cost += cost_per_shot[backend] * self.hardware.shots
                
        return min(total_cost, self.hardware.max_cost_per_experiment)


# Predefined configurations for common use cases
THEORETICAL_ANALYSIS = ExperimentalConfig(
    mode=ExperimentMode.THEORETICAL,
    experiment_name="theoretical_analysis",
    description="Pure theoretical analysis of quantum advantages"
)

SIMULATION_VALIDATION = ExperimentalConfig(
    mode=ExperimentMode.SIMULATION,
    experiment_name="simulation_validation", 
    description="Classical simulation validation of quantum algorithms"
)

HARDWARE_PROOF_OF_CONCEPT = ExperimentalConfig(
    mode=ExperimentMode.HARDWARE,
    experiment_name="hardware_poc",
    description="Proof-of-concept on real quantum hardware",
    hardware=HardwareConfig(
        max_qubits=6,  # Conservative for POC
        shots=512,     # Reduce shots for cost
        use_free_backends_only=True
    )
)

HARDWARE_FULL_EXPERIMENT = ExperimentalConfig(
    mode=ExperimentMode.HARDWARE,
    experiment_name="hardware_full",
    description="Full experimental validation on quantum hardware",
    hardware=HardwareConfig(
        max_qubits=12,
        shots=1024,
        use_free_backends_only=False,
        max_cost_per_experiment=50.0
    )
)


def get_recommended_config(n_qubits: int, budget_usd: float = 0.0) -> ExperimentalConfig:
    """
    Get recommended configuration based on requirements.
    
    Args:
        n_qubits: Required number of qubits
        budget_usd: Available budget for experiments
        
    Returns:
        Recommended experimental configuration
    """
    if n_qubits <= 20 and budget_usd == 0:
        # Free simulation
        config = SIMULATION_VALIDATION.copy()
        config.simulation.max_qubits = n_qubits
        return config
    elif n_qubits <= 12 and budget_usd > 0:
        # Hardware experiments
        if budget_usd < 10:
            config = HARDWARE_PROOF_OF_CONCEPT.copy()
        else:
            config = HARDWARE_FULL_EXPERIMENT.copy()
        config.hardware.max_qubits = n_qubits
        config.hardware.max_cost_per_experiment = budget_usd
        return config
    else:
        # Theoretical analysis only
        return THEORETICAL_ANALYSIS


def validate_experimental_setup(config: ExperimentalConfig) -> bool:
    """
    Validate experimental setup and warn about potential issues.
    
    Args:
        config: Experimental configuration
        
    Returns:
        True if setup is valid
    """
    warnings_dict = config.validate_all()
    
    has_warnings = False
    for mode, warnings_list in warnings_dict.items():
        if warnings_list:
            has_warnings = True
            print(f"⚠️ {mode.upper()} warnings:")
            for warning in warnings_list:
                print(f"  - {warning}")
                
    if config.mode == ExperimentMode.HARDWARE:
        requirements = config.get_hardware_requirements()
        estimated_cost = requirements.get('estimated_cost_usd', 0)
        
        if estimated_cost > 0:
            print(f"💰 Estimated cost: ${estimated_cost:.2f}")
            
        if requirements.get('api_credentials_required'):
            print("🔑 API credentials required for hardware access")
            
    return not has_warnings
