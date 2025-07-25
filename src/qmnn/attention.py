"""
Attention Mechanisms for QMNN

This module contains various attention mechanisms including classical,
quantum-inspired, and hybrid attention implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Dict, Any
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention mechanism.
    
    Provides baseline classical attention for comparison with quantum variants.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final projection
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)  # Average over heads


class QuantumInspiredAttention(nn.Module):
    """
    Quantum-inspired attention mechanism.
    
    Uses quantum-inspired operations without full quantum simulation
    for computational efficiency while maintaining quantum advantages.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, n_qubits: int = 8,
                 dropout: float = 0.1, use_entanglement: bool = True):
        super().__init__()
        
        if n_qubits > 12:
            warnings.warn(f"n_qubits={n_qubits} > 12 may exceed hardware limits")
            n_qubits = min(n_qubits, 12)
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_qubits = n_qubits
        self.head_dim = d_model // n_heads
        self.use_entanglement = use_entanglement
        
        # Classical projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Quantum-inspired parameters
        self.quantum_weights = nn.Parameter(torch.randn(n_heads, n_qubits) * 0.1)
        self.entanglement_strength = nn.Parameter(torch.tensor(0.5))
        self.phase_parameters = nn.Parameter(torch.randn(n_heads, n_qubits) * 0.1)
        
        # Quantum feature mapping
        self.quantum_map = nn.Sequential(
            nn.Linear(self.head_dim, n_qubits),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantum-inspired attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Classical attention scores
        classical_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Quantum-inspired attention computation
        quantum_scores = self._compute_quantum_attention(Q, K)
        
        # Adaptive mixing
        mixing_weight = torch.sigmoid(self.entanglement_strength)
        combined_scores = (1 - mixing_weight) * classical_scores + mixing_weight * quantum_scores
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            combined_scores = combined_scores.masked_fill(mask == 0, -1e9)
            
        # Softmax attention weights
        attention_weights = F.softmax(combined_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final projection
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)
        
    def _compute_quantum_attention(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute quantum-inspired attention scores."""
        batch_size, n_heads, seq_len, head_dim = Q.shape
        
        # Map to quantum feature space
        Q_quantum = self.quantum_map(Q)  # [batch, heads, seq, n_qubits]
        K_quantum = self.quantum_map(K)
        
        # Quantum interference computation
        Q_expanded = Q_quantum.unsqueeze(3)  # [batch, heads, seq, 1, qubits]
        K_expanded = K_quantum.unsqueeze(2)  # [batch, heads, 1, seq, qubits]
        
        # Phase interference
        phase_diff = Q_expanded - K_expanded
        interference = torch.cos(phase_diff + self.phase_parameters.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        
        # Apply quantum weights
        quantum_weights = torch.sigmoid(self.quantum_weights).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        weighted_interference = interference * quantum_weights
        
        # Aggregate over quantum dimensions
        quantum_scores = torch.mean(weighted_interference, dim=-1)
        
        # Add entanglement effects
        if self.use_entanglement:
            quantum_scores = self._apply_entanglement_effects(quantum_scores)
            
        return quantum_scores
        
    def _apply_entanglement_effects(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply entanglement-like correlations to attention scores."""
        batch_size, n_heads, seq_len_q, seq_len_k = scores.shape
        
        # Create entanglement correlation matrix
        entanglement_matrix = torch.ones(seq_len_q, seq_len_k, device=scores.device)
        
        # Apply distance-based entanglement decay
        for i in range(seq_len_q):
            for j in range(seq_len_k):
                distance = abs(i - j)
                entanglement_matrix[i, j] = torch.exp(-distance * 0.1)
                
        # Apply entanglement effects
        entangled_scores = scores * entanglement_matrix.unsqueeze(0).unsqueeze(0)
        
        return entangled_scores


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention that switches between classical and quantum-inspired modes.
    
    Automatically selects the best attention mechanism based on input characteristics.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, n_qubits: int = 8,
                 dropout: float = 0.1, adaptation_threshold: float = 0.5):
        super().__init__()
        
        self.d_model = d_model
        self.adaptation_threshold = adaptation_threshold
        
        # Classical attention
        self.classical_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Quantum-inspired attention
        self.quantum_attention = QuantumInspiredAttention(
            d_model, n_heads, n_qubits, dropout
        )
        
        # Adaptation mechanism
        self.adaptation_network = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through adaptive attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute adaptation score based on input complexity
        input_complexity = self._compute_input_complexity(query, key)
        adaptation_score = self.adaptation_network(input_complexity)
        
        # Choose attention mechanism
        use_quantum = adaptation_score.mean() > self.adaptation_threshold
        
        if use_quantum:
            output, attention_weights = self.quantum_attention(query, key, value, mask)
        else:
            output, attention_weights = self.classical_attention(query, key, value, mask)
            
        return output, attention_weights
        
    def _compute_input_complexity(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute input complexity to guide attention mechanism selection."""
        # Simple complexity measure based on variance and correlation
        query_var = torch.var(query, dim=1)  # [batch, d_model]
        key_var = torch.var(key, dim=1)
        
        # Combine variances as complexity measure
        complexity = (query_var + key_var) / 2
        
        return complexity
