"""
Quantum Watermarking for QMANN

This module implements quantum-inspired watermarking techniques
for protecting datasets and model outputs, based on TabWak methodology.
"""

import torch
import numpy as np
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json


@dataclass
class WatermarkConfig:
    """Configuration for quantum watermarking."""
    watermark_strength: float = 0.1  # Strength of watermark signal
    embedding_ratio: float = 0.05    # Fraction of data to watermark
    quantum_encoding: bool = True     # Use quantum-inspired encoding
    verification_threshold: float = 0.8  # Threshold for watermark detection
    secret_key: Optional[str] = None  # Secret key for watermarking
    preserve_utility: bool = True     # Preserve data utility
    steganographic: bool = True       # Hide watermark presence


class QuantumWatermarkEmbedder:
    """
    Quantum-inspired watermarking for tabular and sequential data.
    
    Based on quantum superposition principles to embed imperceptible
    watermarks that are robust to various attacks.
    """
    
    def __init__(self, config: WatermarkConfig):
        """
        Initialize quantum watermark embedder.
        
        Args:
            config: Watermarking configuration
        """
        self.config = config
        
        # Generate deterministic random state from secret key
        if config.secret_key:
            self.random_state = self._generate_random_state(config.secret_key)
        else:
            self.random_state = np.random.RandomState(42)
            
        # Quantum-inspired parameters
        self.quantum_phases = self._generate_quantum_phases()
        self.entanglement_matrix = self._generate_entanglement_matrix()
        
    def _generate_random_state(self, secret_key: str) -> np.random.RandomState:
        """Generate deterministic random state from secret key."""
        # Hash secret key to get deterministic seed
        hash_object = hashlib.sha256(secret_key.encode())
        seed = int(hash_object.hexdigest()[:8], 16) % (2**32)
        return np.random.RandomState(seed)
        
    def _generate_quantum_phases(self) -> np.ndarray:
        """Generate quantum phase patterns for watermarking."""
        # Generate quantum-inspired phase patterns
        n_phases = 64  # Number of phase patterns
        phases = self.random_state.uniform(0, 2*np.pi, n_phases)
        return phases
        
    def _generate_entanglement_matrix(self) -> np.ndarray:
        """Generate entanglement matrix for quantum correlations."""
        # Create entanglement-inspired correlation matrix
        size = 16
        matrix = self.random_state.randn(size, size)
        # Make it symmetric (like quantum entanglement)
        matrix = (matrix + matrix.T) / 2
        # Normalize
        matrix = matrix / np.linalg.norm(matrix)
        return matrix
        
    def embed_watermark(self, data: torch.Tensor, 
                       watermark_id: str = "qmann_default") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Embed quantum watermark into data.
        
        Args:
            data: Input data tensor [batch_size, seq_len, features] or [batch_size, features]
            watermark_id: Unique identifier for this watermark
            
        Returns:
            Tuple of (watermarked_data, watermark_info)
        """
        original_shape = data.shape
        
        # Flatten data for processing
        if len(data.shape) == 3:
            batch_size, seq_len, features = data.shape
            data_flat = data.view(-1, features)
        elif len(data.shape) == 2:
            batch_size, features = data.shape
            seq_len = 1
            data_flat = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
            
        # Select positions to watermark
        n_positions = int(self.config.embedding_ratio * data_flat.shape[0])
        watermark_positions = self.random_state.choice(
            data_flat.shape[0], n_positions, replace=False
        )
        
        # Create watermarked data copy
        watermarked_data = data_flat.clone()
        
        # Embed quantum watermark
        watermark_signal = self._generate_watermark_signal(
            n_positions, features, watermark_id
        )
        
        # Apply watermark with quantum-inspired modulation
        for i, pos in enumerate(watermark_positions):
            if self.config.quantum_encoding:
                # Quantum-inspired embedding using phase modulation
                phase_idx = i % len(self.quantum_phases)
                phase = self.quantum_phases[phase_idx]
                
                # Modulate watermark signal with quantum phase
                modulated_signal = watermark_signal[i] * np.cos(phase)
                
                # Apply entanglement-inspired correlations
                if i < len(self.entanglement_matrix):
                    correlation = self.entanglement_matrix[i % len(self.entanglement_matrix)]
                    modulated_signal = modulated_signal * correlation[:features]
                    
            else:
                modulated_signal = watermark_signal[i]
                
            # Embed watermark while preserving data utility
            if self.config.preserve_utility:
                # Adaptive embedding based on data magnitude
                data_magnitude = torch.abs(watermarked_data[pos]).mean()
                adaptive_strength = self.config.watermark_strength * data_magnitude
                watermarked_data[pos] += adaptive_strength * torch.tensor(modulated_signal, dtype=data.dtype)
            else:
                watermarked_data[pos] += self.config.watermark_strength * torch.tensor(modulated_signal, dtype=data.dtype)
                
        # Reshape back to original shape
        watermarked_data = watermarked_data.view(original_shape)
        
        # Create watermark info
        watermark_info = {
            'watermark_id': watermark_id,
            'positions': watermark_positions.tolist(),
            'signal_hash': self._hash_signal(watermark_signal),
            'embedding_strength': self.config.watermark_strength,
            'quantum_encoding': self.config.quantum_encoding,
            'data_shape': original_shape,
            'n_watermarked_positions': n_positions
        }
        
        return watermarked_data, watermark_info
        
    def _generate_watermark_signal(self, n_positions: int, features: int, 
                                  watermark_id: str) -> np.ndarray:
        """Generate watermark signal based on ID and quantum patterns."""
        # Create deterministic signal from watermark ID
        id_hash = hashlib.sha256(watermark_id.encode()).hexdigest()
        id_seed = int(id_hash[:8], 16) % (2**32)
        
        # Generate signal with ID-specific randomness
        signal_rng = np.random.RandomState(id_seed)
        signal = signal_rng.randn(n_positions, features)
        
        # Apply quantum-inspired transformations
        if self.config.quantum_encoding:
            # Apply quantum phase patterns
            for i in range(n_positions):
                phase_idx = i % len(self.quantum_phases)
                phase = self.quantum_phases[phase_idx]
                
                # Quantum-inspired amplitude modulation
                signal[i] *= np.cos(phase) + 1j * np.sin(phase)
                signal[i] = np.real(signal[i])  # Take real part
                
        # Normalize signal
        signal = signal / np.linalg.norm(signal, axis=1, keepdims=True)
        
        return signal
        
    def _hash_signal(self, signal: np.ndarray) -> str:
        """Create hash of watermark signal for verification."""
        signal_bytes = signal.tobytes()
        return hashlib.sha256(signal_bytes).hexdigest()[:16]
        
    def detect_watermark(self, data: torch.Tensor, 
                        watermark_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect watermark in potentially modified data.
        
        Args:
            data: Data to check for watermark
            watermark_info: Information about embedded watermark
            
        Returns:
            Detection results
        """
        # Flatten data
        original_shape = watermark_info['data_shape']
        if len(data.shape) == 3:
            data_flat = data.view(-1, data.shape[-1])
        elif len(data.shape) == 2:
            data_flat = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
            
        # Extract watermark positions
        positions = watermark_info['positions']
        
        # Regenerate expected watermark signal
        expected_signal = self._generate_watermark_signal(
            len(positions), data_flat.shape[1], watermark_info['watermark_id']
        )
        
        # Extract signal from watermarked positions
        extracted_signals = []
        for i, pos in enumerate(positions):
            if pos < data_flat.shape[0]:
                extracted_signals.append(data_flat[pos].numpy())
                
        if not extracted_signals:
            return {
                'watermark_detected': False,
                'confidence': 0.0,
                'correlation': 0.0,
                'error': 'No valid positions found'
            }
            
        extracted_signals = np.array(extracted_signals)
        
        # Calculate correlation with expected signal
        correlation = self._calculate_correlation(extracted_signals, expected_signal)
        
        # Determine if watermark is detected
        watermark_detected = correlation > self.config.verification_threshold
        confidence = min(correlation / self.config.verification_threshold, 1.0)
        
        return {
            'watermark_detected': watermark_detected,
            'confidence': confidence,
            'correlation': correlation,
            'threshold': self.config.verification_threshold,
            'positions_checked': len(positions),
            'signal_hash_match': self._hash_signal(expected_signal) == watermark_info['signal_hash']
        }
        
    def _calculate_correlation(self, extracted: np.ndarray, 
                              expected: np.ndarray) -> float:
        """Calculate correlation between extracted and expected signals."""
        if extracted.shape != expected.shape:
            # Resize if shapes don't match
            min_rows = min(extracted.shape[0], expected.shape[0])
            min_cols = min(extracted.shape[1], expected.shape[1])
            extracted = extracted[:min_rows, :min_cols]
            expected = expected[:min_rows, :min_cols]
            
        # Flatten for correlation calculation
        extracted_flat = extracted.flatten()
        expected_flat = expected.flatten()
        
        # Calculate Pearson correlation
        if np.std(extracted_flat) == 0 or np.std(expected_flat) == 0:
            return 0.0
            
        correlation = np.corrcoef(extracted_flat, expected_flat)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            return 0.0
            
        return abs(correlation)  # Use absolute correlation
        
    def remove_watermark(self, data: torch.Tensor, 
                        watermark_info: Dict[str, Any]) -> torch.Tensor:
        """
        Attempt to remove watermark from data.
        
        Args:
            data: Watermarked data
            watermark_info: Watermark information
            
        Returns:
            Data with watermark removed (best effort)
        """
        warnings.warn("Watermark removal is not guaranteed to be complete")
        
        # Flatten data
        original_shape = data.shape
        if len(data.shape) == 3:
            data_flat = data.view(-1, data.shape[-1])
        elif len(data.shape) == 2:
            data_flat = data.clone()
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
            
        # Regenerate watermark signal
        positions = watermark_info['positions']
        watermark_signal = self._generate_watermark_signal(
            len(positions), data_flat.shape[1], watermark_info['watermark_id']
        )
        
        # Remove watermark (best effort)
        for i, pos in enumerate(positions):
            if pos < data_flat.shape[0] and i < len(watermark_signal):
                # Estimate and subtract watermark
                estimated_watermark = watermark_info['embedding_strength'] * torch.tensor(
                    watermark_signal[i], dtype=data.dtype
                )
                data_flat[pos] -= estimated_watermark
                
        return data_flat.view(original_shape)


def prepare_dataset_with_watermark(dataset: torch.Tensor, 
                                  watermark_config: Optional[WatermarkConfig] = None,
                                  watermark_id: str = "qmann_dataset") -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Prepare dataset with quantum watermarking.
    
    Args:
        dataset: Input dataset
        watermark_config: Watermarking configuration
        watermark_id: Unique watermark identifier
        
    Returns:
        Tuple of (watermarked_dataset, watermark_metadata)
    """
    if watermark_config is None:
        watermark_config = WatermarkConfig()
        
    embedder = QuantumWatermarkEmbedder(watermark_config)
    watermarked_data, watermark_info = embedder.embed_watermark(dataset, watermark_id)
    
    metadata = {
        'watermark_info': watermark_info,
        'config': {
            'watermark_strength': watermark_config.watermark_strength,
            'embedding_ratio': watermark_config.embedding_ratio,
            'quantum_encoding': watermark_config.quantum_encoding,
            'verification_threshold': watermark_config.verification_threshold,
            'preserve_utility': watermark_config.preserve_utility,
            'steganographic': watermark_config.steganographic
        },
        'original_shape': dataset.shape,
        'watermarked_shape': watermarked_data.shape
    }
    
    return watermarked_data, metadata


def verify_dataset_watermark(dataset: torch.Tensor, 
                            watermark_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify watermark in dataset.
    
    Args:
        dataset: Dataset to verify
        watermark_metadata: Watermark metadata from preparation
        
    Returns:
        Verification results
    """
    config = WatermarkConfig(**watermark_metadata['config'])
    embedder = QuantumWatermarkEmbedder(config)
    
    return embedder.detect_watermark(dataset, watermark_metadata['watermark_info'])


def save_watermark_metadata(metadata: Dict[str, Any], filepath: str):
    """Save watermark metadata to file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            serializable_metadata[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_metadata[key][k] = v.tolist()
                else:
                    serializable_metadata[key][k] = v
        else:
            serializable_metadata[key] = value
            
    with open(filepath, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)


def load_watermark_metadata(filepath: str) -> Dict[str, Any]:
    """Load watermark metadata from file."""
    with open(filepath, 'r') as f:
        metadata = json.load(f)
        
    # Convert lists back to numpy arrays where needed
    if 'watermark_info' in metadata and 'positions' in metadata['watermark_info']:
        metadata['watermark_info']['positions'] = np.array(metadata['watermark_info']['positions'])
        
    return metadata
