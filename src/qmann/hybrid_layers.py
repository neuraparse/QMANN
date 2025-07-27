"""
Hybrid Quantum-Classical Layers for QMANN

This module contains hybrid layers that seamlessly integrate quantum
and classical processing components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import warnings

from .attention import MultiHeadAttention, QuantumInspiredAttention, AdaptiveAttention
from .classical_controller import ClassicalEncoder, ClassicalDecoder


class HybridTransformerLayer(nn.Module):
    """
    Hybrid transformer layer combining classical and quantum-inspired attention.
    
    Provides seamless integration between classical transformer operations
    and quantum-enhanced attention mechanisms.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1, attention_type: str = "adaptive",
                 n_qubits: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Attention mechanism selection
        if attention_type == "classical":
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == "quantum":
            self.attention = QuantumInspiredAttention(d_model, n_heads, n_qubits, dropout)
        elif attention_type == "adaptive":
            self.attention = AdaptiveAttention(d_model, n_heads, n_qubits, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Quantum-classical fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through hybrid transformer layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        
        # Apply fusion gate for quantum-classical integration
        fusion_weights = self.fusion_gate(x)
        fused_attn = attn_output * fusion_weights
        
        x = self.norm1(x + self.dropout(fused_attn))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        if return_attention:
            return x, attention_weights
        return x


class QuantumClassicalFusion(nn.Module):
    """
    Fusion layer for combining quantum and classical representations.
    
    Implements various fusion strategies for optimal information integration.
    """
    
    def __init__(self, quantum_dim: int, classical_dim: int, output_dim: int,
                 fusion_type: str = "gated", dropout: float = 0.1):
        super().__init__()
        
        self.quantum_dim = quantum_dim
        self.classical_dim = classical_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concatenation":
            self.fusion_layer = nn.Sequential(
                nn.Linear(quantum_dim + classical_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == "gated":
            # Gated fusion mechanism
            self.quantum_gate = nn.Sequential(
                nn.Linear(quantum_dim + classical_dim, quantum_dim),
                nn.Sigmoid()
            )
            self.classical_gate = nn.Sequential(
                nn.Linear(quantum_dim + classical_dim, classical_dim),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(quantum_dim + classical_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == "attention":
            # Attention-based fusion
            self.fusion_attention = MultiHeadAttention(
                d_model=max(quantum_dim, classical_dim),
                n_heads=min(8, max(quantum_dim, classical_dim) // 64),
                dropout=dropout
            )
            # Projection layers to match dimensions
            self.quantum_proj = nn.Linear(quantum_dim, max(quantum_dim, classical_dim))
            self.classical_proj = nn.Linear(classical_dim, max(quantum_dim, classical_dim))
            self.output_proj = nn.Linear(max(quantum_dim, classical_dim), output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
    def forward(self, quantum_repr: torch.Tensor, classical_repr: torch.Tensor) -> torch.Tensor:
        """
        Fuse quantum and classical representations.
        
        Args:
            quantum_repr: Quantum representation [batch_size, seq_len, quantum_dim]
            classical_repr: Classical representation [batch_size, seq_len, classical_dim]
            
        Returns:
            Fused representation [batch_size, seq_len, output_dim]
        """
        if self.fusion_type == "concatenation":
            # Simple concatenation
            combined = torch.cat([quantum_repr, classical_repr], dim=-1)
            return self.fusion_layer(combined)
            
        elif self.fusion_type == "gated":
            # Gated fusion
            combined = torch.cat([quantum_repr, classical_repr], dim=-1)
            
            quantum_gate = self.quantum_gate(combined)
            classical_gate = self.classical_gate(combined)
            
            gated_quantum = quantum_repr * quantum_gate
            gated_classical = classical_repr * classical_gate
            
            fused = torch.cat([gated_quantum, gated_classical], dim=-1)
            return self.fusion_layer(fused)
            
        elif self.fusion_type == "attention":
            # Attention-based fusion
            quantum_proj = self.quantum_proj(quantum_repr)
            classical_proj = self.classical_proj(classical_repr)
            
            # Use quantum as query, classical as key/value
            fused, _ = self.fusion_attention(quantum_proj, classical_proj, classical_proj)
            
            return self.output_proj(fused)


class AdaptiveHybridLayer(nn.Module):
    """
    Adaptive hybrid layer that dynamically adjusts quantum-classical balance.
    
    Automatically determines the optimal mix of quantum and classical processing
    based on input characteristics and task requirements.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_qubits: int = 8, adaptation_steps: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.adaptation_steps = adaptation_steps
        
        # Classical processing path
        self.classical_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Quantum-inspired processing path
        self.quantum_path = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Tanh(),  # Quantum normalization
            nn.Linear(n_qubits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Adaptation mechanism
        self.adaptation_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Performance tracking
        self.register_buffer('classical_performance', torch.tensor(0.5))
        self.register_buffer('quantum_performance', torch.tensor(0.5))
        self.register_buffer('adaptation_history', torch.zeros(adaptation_steps))
        self.register_buffer('step_counter', torch.tensor(0))
        
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through adaptive hybrid layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            target: Optional target for performance tracking
            
        Returns:
            Dictionary with outputs and adaptation information
        """
        # Compute both paths
        classical_output = self.classical_path(x)
        quantum_output = self.quantum_path(x)
        
        # Compute adaptation weight
        adaptation_weight = self.adaptation_network(x.mean(dim=1))  # [batch_size, 1]
        
        # Adaptive combination
        combined_output = (
            (1 - adaptation_weight.unsqueeze(1)) * classical_output +
            adaptation_weight.unsqueeze(1) * quantum_output
        )
        
        # Update performance tracking if target is provided
        if target is not None and self.training:
            self._update_performance_tracking(
                classical_output, quantum_output, target, adaptation_weight
            )
            
        return {
            'output': combined_output,
            'classical_output': classical_output,
            'quantum_output': quantum_output,
            'adaptation_weight': adaptation_weight,
            'classical_performance': self.classical_performance,
            'quantum_performance': self.quantum_performance
        }
        
    def _update_performance_tracking(self, classical_out: torch.Tensor, 
                                   quantum_out: torch.Tensor,
                                   target: torch.Tensor, 
                                   adaptation_weight: torch.Tensor) -> None:
        """Update performance tracking for adaptation."""
        with torch.no_grad():
            # Compute losses
            classical_loss = F.mse_loss(classical_out, target)
            quantum_loss = F.mse_loss(quantum_out, target)
            
            # Update performance estimates with exponential moving average
            alpha = 0.1
            self.classical_performance = (1 - alpha) * self.classical_performance + alpha * (1 / (1 + classical_loss))
            self.quantum_performance = (1 - alpha) * self.quantum_performance + alpha * (1 / (1 + quantum_loss))
            
            # Update adaptation history
            current_step = self.step_counter % self.adaptation_steps
            self.adaptation_history[current_step] = adaptation_weight.mean()
            self.step_counter += 1
            
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        return {
            'classical_performance': float(self.classical_performance),
            'quantum_performance': float(self.quantum_performance),
            'current_adaptation_weight': float(self.adaptation_history[self.step_counter % self.adaptation_steps]),
            'avg_adaptation_weight': float(self.adaptation_history.mean()),
            'adaptation_variance': float(self.adaptation_history.var())
        }


class HybridResidualBlock(nn.Module):
    """
    Residual block with hybrid quantum-classical processing.
    
    Implements residual connections for stable training of deep hybrid networks.
    """
    
    def __init__(self, dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.n_qubits = n_qubits
        
        # Hybrid processing layers
        self.hybrid_layer1 = AdaptiveHybridLayer(dim, dim, dim, n_qubits)
        self.hybrid_layer2 = AdaptiveHybridLayer(dim, dim, dim, n_qubits)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid residual block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # First hybrid layer with residual connection
        out1 = self.hybrid_layer1(x)['output']
        x = self.norm1(x + self.dropout(out1))
        
        # Second hybrid layer with residual connection
        out2 = self.hybrid_layer2(x)['output']
        x = self.norm2(x + self.dropout(out2))
        
        return x
